#![allow(static_mut_refs)]

use core::f32;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex, mpsc};
use std::{fs, thread};

use eframe::{App, Frame, NativeOptions};
use egui::epaint::PathStroke;
use egui::load::SizedTexture;
use egui::{
    Button, CentralPanel, Color32, ColorImage, DragValue, Grid, Id, Image, Label, LayerId, Margin,
    Painter, Pos2, Rect, RichText, SelectableLabel, Sense, Slider, Stroke, TextEdit, TextureHandle,
    TextureOptions, Ui, Widget, Window, emath, epaint, pos2,
};

use enum_map::{Enum, EnumMap};
use eye::colorconvert::Device;
use eye::hal::format::PixelFormat;
use eye::hal::traits::{Context as _, Device as _, Stream as _};
use eye::hal::{Error, ErrorKind, PlatformContext};

use anyhow::{Result, bail};
use inotify::{Inotify, WatchMask};
use serde::{Deserialize, Serialize};

struct UVCPlayer {
    texture: TextureHandle,
    rects: Vec<Rect>,
    rect_begin: Option<Pos2>,

    rect_motion: Option<Pos2>,
    ratio: Calibration,
    profiles: BTreeMap<PathBuf, MicroscopeRatio>,
    active_profile: Option<PathBuf>,
    new_profile_name: String,

    show_labels: bool,

    waiting: Arc<Mutex<bool>>,

    // Camera selection
    available_cameras: Arc<Mutex<Vec<CameraInfo>>>,
    selected_camera_uri: Arc<Mutex<Option<String>>>,
    camera_cmd_tx: mpsc::Sender<CameraCommand>,
}

#[derive(Clone, Debug)]
struct CameraInfo {
    uri: String,
    name: String,
}

#[derive(Debug)]
enum CameraCommand {
    Refresh,
    Select(String),
}

#[derive(Default, Clone, Copy, Serialize, Deserialize)]
struct MicroscopeRatio {
    /// Microns per pixel
    map: EnumMap<Magnification, f64>,
}

#[derive(Default, Clone, Copy)]
struct Calibration {
    result: MicroscopeRatio,
    active: Option<Magnification>,
    calibrating: bool,
}

impl MicroscopeRatio {
    pub fn calibrate(&mut self, active: Magnification, pixels: f64) {
        let ratio = active.default_cal_len() / pixels;
        self.map[active] = ratio;
    }
}

#[derive(Clone, Copy, Enum, PartialEq, Eq, Serialize, Deserialize)]
enum Magnification {
    X4,
    X10,
    X40,
    X100,
}

impl Magnification {
    pub fn button_text(self) -> &'static str {
        match self {
            Self::X4 => "4x 1mm",
            Self::X10 => "10x 0.7mm",
            Self::X40 => "40x 0.15mm",
            Self::X100 => "100X 0.05mm",
        }
    }
    pub fn default_cal_len(&self) -> f64 {
        match &self {
            Self::X4 => 1e3,
            Self::X10 => 0.7e3,
            Self::X40 => 0.15e3,
            Self::X100 => 0.05e3,
        }
    }
}

impl Calibration {
    pub fn from_px(&self, px: f64) -> f64 {
        let active = self.active.unwrap();
        let rate = self.result.map[active];
        px * rate
    }
}

impl Widget for &mut Calibration {
    fn ui(self, ui: &mut Ui) -> egui::Response {
        ui.vertical(|ui| {
            ui.add_space(20.);
            ui.add(Label::new(
                RichText::new("calibration").color(Color32::WHITE.gamma_multiply(0.9)),
            ));
            ui.add_space(10.);

            let result = self.result;
            for (target, val) in result.map.iter() {
                let mut btn = Button::new(target.button_text());

                if *val > 0. {
                    btn = btn.fill(Color32::DARK_GREEN.gamma_multiply(0.8));
                } else {
                    // grey, default
                }

                if let Some(active) = self.active {
                    if target == active {
                        btn = btn.fill(Color32::from_rgb(122, 104, 1));
                    }
                }

                if ui.add(btn).clicked() {
                    // Ig clearing this is better
                    self.calibrating = false;
                    self.active = Some(target.clone());
                }
            }

            if self.active.is_some() {
                ui.add_space(20.);
                let mut btn = ui.add(Button::new("calibrate"));
                if self.calibrating {
                    btn = btn.highlight();
                }
                if btn.clicked() {
                    if self.calibrating {
                        self.calibrating = false;
                    } else {
                        self.calibrating = true;
                    }
                }

                ui.add(Label::new(RichText::new(format!(
                    "{} µm/px",
                    self.result.map[self.active.unwrap()]
                ))));
            }
        });

        ui.response()
    }
}

struct StreamD<'a> {
    st: eye::hal::platform::Stream<'a>,
    d: [usize; 2],
}

fn list_cameras() -> Vec<CameraInfo> {
    let mut cameras = Vec::new();
    if let Some(ctx) = PlatformContext::all().next() {
        if let Ok(dev_descrs) = ctx.devices() {
            for dev in dev_descrs {
                cameras.push(CameraInfo {
                    uri: dev.uri.clone(),
                    name: dev.uri.clone(), // Use URI as name, could parse for friendlier name
                });
            }
        }
    }
    cameras
}

fn find_stream_for_camera<'a>(uri: &str) -> anyhow::Result<Option<StreamD<'a>>> {
    let ctx = if let Some(ctx) = PlatformContext::all().next() {
        ctx
    } else {
        return Ok(None);
    };

    let dev = ctx.open_device(uri)?;
    let dev = Device::new(dev)?;
    dbg!(&dev.streams());
    let maxxed = dev
        .streams()?
        .into_iter()
        .filter(|x| x.pixfmt == PixelFormat::Rgb(24))
        .filter(|x| x.width <= 2560)
        .max_by_key(|d| d.width as u128 * d.height as u128 / d.interval.as_millis());

    let stream_descr = match maxxed {
        Some(s) => s,
        None => return Ok(None),
    };
    let dimensions = [stream_descr.width as usize, stream_descr.height as usize];
    println!("Selected stream:\n{:?}", stream_descr);

    let stream = dev.start_stream(&stream_descr)?;

    Ok(Some(StreamD {
        st: stream,
        d: dimensions,
    }))
}

fn find_stream<'a>() -> anyhow::Result<Option<StreamD<'a>>> {
    let ctx = if let Some(ctx) = PlatformContext::all().next() {
        ctx
    } else {
        return Ok(None);
    };

    // Create a list of valid capture devices in the system.
    let dev_descrs = ctx.devices()?;
    dbg!(&dev_descrs);

    if dev_descrs.len() == 0 {
        return Ok(None);
    }
    // Print the supported formats for each device.
    let dev = ctx.open_device(&dev_descrs[0].uri)?;
    // let dev = ctx.open_device("v4l:///dev/video")?;
    let dev = Device::new(dev)?;
    dbg!(&dev.streams());
    let maxxed = dev
        .streams()?
        .into_iter()
        .filter(|x| x.pixfmt == PixelFormat::Rgb(24))
        .filter(|x| x.width <= 2560)
        .max_by_key(|d| d.width as u128 * d.height as u128 / d.interval.as_millis())
        .unwrap();

    let stream_descr = maxxed;
    let dimensions = [stream_descr.width as usize, stream_descr.height as usize];
    println!("Selected stream:\n{:?}", stream_descr);

    let stream = dev.start_stream(&stream_descr)?;

    Ok(Some(StreamD {
        st: stream,
        d: dimensions,
    }))
}

fn main() -> Result<()> {
    let (frame_tx, frame_rx) = mpsc::channel::<(Vec<u8>, [usize; 2])>();
    let (camera_cmd_tx, camera_cmd_rx) = mpsc::channel::<CameraCommand>();
    
    // Shared state for camera list (updated by camera thread)
    let available_cameras: Arc<Mutex<Vec<CameraInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let selected_camera_uri: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    
    let cameras_for_thread = available_cameras.clone();
    let selected_for_thread = selected_camera_uri.clone();

    // Single thread that owns the stream and handles ALL camera operations
    thread::spawn(move || {
        let mut current_stream: Option<StreamD> = None;
        
        // Initial camera list
        let initial_cameras = list_cameras();
        if let Ok(mut cams) = cameras_for_thread.lock() {
            *cams = initial_cameras;
        }
        
        // Try to open first available camera on startup
        if let Ok(Some(s)) = find_stream() {
            current_stream = Some(s);
        }
        
        loop {
            // Check for commands (non-blocking)
            while let Ok(cmd) = camera_cmd_rx.try_recv() {
                match cmd {
                    CameraCommand::Refresh => {
                        println!("Refreshing camera list...");
                        // Must drop stream before enumerating devices
                        let had_stream = current_stream.is_some();
                        let old_uri = selected_for_thread.lock().ok().and_then(|s| s.clone());
                        current_stream = None;
                        
                        let cameras = list_cameras();
                        if let Ok(mut cams) = cameras_for_thread.lock() {
                            *cams = cameras;
                        }
                        
                        // Re-open previous camera if it still exists
                        if had_stream {
                            if let Some(uri) = old_uri {
                                if let Ok(Some(s)) = find_stream_for_camera(&uri) {
                                    current_stream = Some(s);
                                }
                            }
                        }
                    }
                    CameraCommand::Select(uri) => {
                        println!("Switching to camera: {}", uri);
                        // Drop old stream first
                        current_stream = None;
                        
                        // Update selected URI
                        if let Ok(mut sel) = selected_for_thread.lock() {
                            *sel = Some(uri.clone());
                        }
                        
                        // Open new stream
                        match find_stream_for_camera(&uri) {
                            Ok(Some(s)) => {
                                current_stream = Some(s);
                                println!("Successfully opened camera");
                            }
                            Ok(None) => {
                                println!("No compatible stream found for camera");
                            }
                            Err(e) => {
                                println!("Error opening camera: {:?}", e);
                            }
                        }
                    }
                }
            }
            
            // Read frame from current stream
            if let Some(ref mut stream) = current_stream {
                let buf: Option<std::result::Result<&[u8], _>> = stream.st.next();
                if let Some(Ok(buf)) = buf {
                    // Send frame data to render thread
                    let _ = frame_tx.send((buf.to_vec(), stream.d));
                } else {
                    println!("stream ended");
                    current_stream = None;
                }
            } else {
                // No stream, sleep a bit to avoid busy loop
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    });

    let _ = eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|ctx| {
            let mut app = UVCPlayer {
                texture: ctx.egui_ctx.load_texture(
                    "vid",
                    ColorImage::new([1, 1], Color32::BLACK),
                    Default::default(),
                ),
                rect_begin: None,
                rect_motion: None,
                rects: Default::default(),
                ratio: Default::default(),
                profiles: Default::default(),
                active_profile: None,
                new_profile_name: "0.5x".to_owned(),
                show_labels: true,
                waiting: Arc::new(Mutex::new(true)),
                available_cameras,
                selected_camera_uri,
                camera_cmd_tx,
            };

            app.load_profiles()?;

            let mut txt = app.texture.clone();
            let ctx = ctx.egui_ctx.clone();
            let waiting = app.waiting.clone();

            thread::spawn(move || {
                loop {
                    // Receive frames from the camera thread
                    if let Ok((buf, dimensions)) = frame_rx.recv() {
                        let mut k = waiting.lock().unwrap();
                        *k = false;
                        drop(k);
                        
                        txt.set(
                            ColorImage::from_rgb(dimensions, &buf),
                            TextureOptions::default(),
                        );
                        ctx.request_repaint();
                    } else {
                        // Channel closed or error
                        let mut k = waiting.lock().unwrap();
                        *k = true;
                        drop(k);
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                }
            });

            Result::Ok(Box::new(app))
        }),
    );

    Ok(())
}

impl UVCPlayer {
    const PROFILE_DIR: &str = "./profiles";
    pub fn path_default_profile() -> PathBuf {
        PathBuf::from_str("./profiles/default.json").unwrap()
    }
    pub fn load_profiles(&mut self) -> Result<()> {
        // scan for profiles
        let profile_path = Self::PROFILE_DIR;
        let rd = std::fs::read_dir(profile_path);
        if rd.is_err() {
            std::fs::create_dir(profile_path)?;
        } else {
            let rd = rd?;
            for entry in rd {
                let ent = entry?;
                self.profile_load(ent.path())?;
            }
        }
        let defprof = Self::path_default_profile();
        if !self.profiles.contains_key(&defprof) {
            self.profiles.insert(defprof.clone(), Default::default());
        }

        if self.active_profile.is_none() {
            self.active_profile = Some(defprof);
        }
        self.sync_from_profile();

        Ok(())
    }
    pub fn profile_load(&mut self, p: PathBuf) -> Result<()> {
        let fd = std::fs::read_to_string(&p)?;
        let data = serde_json::from_str(&fd)?;
        self.profiles.insert(p, data);
        Ok(())
    }
    pub fn dump_profile(&self) -> Result<()> {
        if let Some(ref pf) = self.active_profile {
            if let Some(v) = self.profiles.get(pf) {
                let fd = fs::File::create(&pf)?;
                serde_json::to_writer_pretty(fd, v)?;
            }
        }
        Ok(())
    }
    pub fn sync_to_profile(&mut self) {
        let active = self.active_profile.as_ref().unwrap();
        self.profiles.insert(active.clone(), self.ratio.result);
    }
    pub fn sync_from_profile(&mut self) {
        let active = self.active_profile.as_ref().unwrap();
        self.ratio.result = self.profiles[active];
    }
    pub fn make_profile(&mut self) -> Result<()> {
        let new_p = PathBuf::from_iter(&[
            Self::PROFILE_DIR,
            &(self.new_profile_name.clone() + ".json"),
        ]);
        self.active_profile = Some(new_p);
        self.sync_to_profile();
        self.dump_profile()?;
        self.load_profiles()?;
        Ok(())
    }
}

impl App for UVCPlayer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::SidePanel::new(egui::panel::Side::Right, "rpanel").show(ctx, |ui| {
            ui.add_space(10.);

            // Camera selection UI
            ui.add(Label::new(
                RichText::new("Cameras").color(Color32::WHITE.gamma_multiply(0.9)),
            ));
            ui.add_space(5.);

            if ui.button("⟳ Refresh").clicked() {
                let _ = self.camera_cmd_tx.send(CameraCommand::Refresh);
            }
            ui.add_space(5.);

            // Clone data to avoid holding locks during UI rendering
            let cameras: Vec<CameraInfo> = self.available_cameras.lock()
                .map(|c| c.clone())
                .unwrap_or_default();
            let selected_uri: Option<String> = self.selected_camera_uri.lock()
                .ok()
                .and_then(|s| s.clone());

            if cameras.is_empty() {
                ui.label("No cameras found");
            } else {
                let mut camera_to_select: Option<String> = None;
                for cam in cameras.iter() {
                    let is_selected = selected_uri.as_ref().map_or(false, |u| u == &cam.uri);
                    
                    // Extract device name from URI for display
                    let display_name = cam.uri.split('/').last().unwrap_or(&cam.uri);
                    
                    if ui.selectable_label(is_selected, display_name).clicked() {
                        camera_to_select = Some(cam.uri.clone());
                    }
                }
                
                if let Some(uri) = camera_to_select {
                    let _ = self.camera_cmd_tx.send(CameraCommand::Select(uri));
                }
            }

            ui.add_space(15.);
            ui.separator();
            ui.add_space(10.);

            ui.add(Label::new(
                RichText::new("Profiles").color(Color32::WHITE.gamma_multiply(0.9)),
            ));
            ui.add_space(5.);

            for (pb, data) in &self.profiles {
                let lb = ui.selectable_label(
                    self.active_profile.as_ref().map_or(false, |v| v == pb),
                    pb.file_stem().unwrap().to_str().unwrap(),
                );
                if lb.clicked() {
                    self.active_profile = Some(pb.clone());
                }
            }

            ui.add_space(6.);
            ui.horizontal(|ui| {
                let text = TextEdit::singleline(&mut self.new_profile_name).desired_width(60.);
                ui.add(text.char_limit(10));
                if ui.button("make profile").clicked() {
                    self.make_profile().unwrap();
                }
            });

            self.ratio.ui(ui);

            ui.add_space(20.);
            ui.add(Label::new(
                "press ESC to clear boxes. \npress Z to hide labels",
            ));
        });
        CentralPanel::default().show(ctx, |ui| {
            if self.waiting.try_lock().map_or(true, |x| *x) {
                ui.centered_and_justified(|ui| ui.label("waiting for device"))
                    .response
            } else {
                let response = ui.add(
                    Image::new(SizedTexture::from_handle(&self.texture))
                        .maintain_aspect_ratio(true)
                        .shrink_to_fit(),
                );

                let pt = ui.painter();
                let sense = response.interact(Sense::all());

                ui.input(|k| {
                    if k.key_pressed(egui::Key::Escape) {
                        self.rects.clear();
                    }
                    if k.key_pressed(egui::Key::Z) {
                        self.show_labels = !self.show_labels;
                    }
                });

                if sense.drag_started() {
                    if let Some(p) = sense.interact_pointer_pos() {
                        self.rect_begin = Some(p);
                    }
                }

                if sense.dragged() {
                    let _ = sense.drag_motion();
                    if let Some(p) = sense.interact_pointer_pos() {
                        self.rect_motion = Some(p);
                    }
                }
                let mut appeneded = Vec::new();
                if let Some(bg) = self.rect_begin {
                    if let Some(mv) = self.rect_motion {
                        let moving_rect = Rect::from_points(&[bg, mv]);
                        appeneded = vec![moving_rect];

                        if self.ratio.calibrating {
                            let len = moving_rect.width();
                            self.ratio
                                .result
                                .calibrate(self.ratio.active.unwrap(), len as f64);
                        }

                        if sense.drag_stopped() {
                            self.rects.push(moving_rect);
                            self.rect_begin = None;
                            self.rect_motion = None;
                        }
                    }
                }
                if sense.drag_stopped() {
                    self.ratio.calibrating = false;
                    self.sync_to_profile();
                    let rex = self.dump_profile();
                    if rex.is_err() {
                        dbg!(&rex);
                    }
                }

                let all_rects = self.rects.iter();
                for rect in all_rects.clone() {
                    let inside_width = 3.;
                    pt.add(epaint::RectShape::new(
                        *rect,
                        0,
                        Color32::TRANSPARENT,
                        Stroke::new(inside_width, Color32::WHITE.gamma_multiply(0.5)),
                        egui::StrokeKind::Outside,
                    ));
                    // let out_rect = rect.expand(inside_width);
                    // pt.add(epaint::RectShape::new(
                    //     out_rect,
                    //     0,
                    //     Color32::TRANSPARENT,
                    //     Stroke::new(2., Color32::WHITE.gamma_multiply(0.5)),
                    //     egui::StrokeKind::Outside,
                    // ));
                }

                let dist_label = |px: f32| {
                    if let Some(a) = self.ratio.active {
                        let n = self.ratio.from_px(px as f64);
                        format!("{:.2}µm", n)
                    } else {
                        format!("{}px", px.round())
                    }
                };

                for rect in all_rects {
                    if self.show_labels {
                        let rect_lb_x = Rect::EVERYTHING
                            .with_min_x(rect.left() - 100.)
                            .with_max_x(rect.right() + 100.)
                            .with_min_y(rect.top() - 60.)
                            .with_max_y(rect.top() - 8.);
                        let wd = dist_label(rect.width());
                        let lb = RichText::new(wd)
                            .color(Color32::WHITE)
                            .size(40.)
                            .background_color(Color32::BLACK.gamma_multiply(0.2));
                        let lb = Label::new(lb);
                        let rect_lb_y = Rect::EVERYTHING
                            .with_min_x(rect.right() - 60.)
                            .with_min_y(rect.top())
                            .with_max_y(rect.bottom())
                            .with_max_x(rect.right() + 200.);

                        ui.put(rect_lb_x, lb);
                        let ht = dist_label(rect.height());
                        let lb = RichText::new(ht)
                            .color(Color32::WHITE)
                            .size(40.)
                            .background_color(Color32::BLACK.gamma_multiply(0.2));
                        let lb = Label::new(lb);

                        ui.put(rect_lb_y, lb);
                    }
                }

                if let Some(pos) = ctx.pointer_latest_pos() {
                    ui.painter().add(epaint::PathShape::line(
                        vec![pos2(pos.x, 0.), pos2(pos.x, ctx.screen_rect().height())],
                        PathStroke::new(3., Color32::WHITE),
                    ));
                    ui.painter().add(epaint::PathShape::line(
                        vec![pos2(0., pos.y), pos2(ctx.screen_rect().width(), pos.y)],
                        PathStroke::new(3., Color32::WHITE),
                    ));
                }
                response
            }
        });
    }
}
