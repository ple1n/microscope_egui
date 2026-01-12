#![allow(static_mut_refs)]

use core::f32;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{fs, time::Duration};

use std::sync::Arc;
use tokio::sync::{Mutex as TokioMutex, RwLock, mpsc};

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

    waiting: Arc<TokioMutex<Option<StreamD>>>,

    // Camera selection
    available_cameras: Arc<RwLock<Arc<Vec<CameraInfo>>>>,
    selected_camera_uri: Arc<RwLock<Option<String>>>,
    selected_stream: Option<(u32, u32, u32)>, // (width, height, fps)
    camera_cmd_tx: mpsc::UnboundedSender<CameraCommand>,
    verbose_camera_ui: bool,
}

#[derive(Clone, Debug)]
struct StreamInfo {
    width: u32,
    height: u32,
    fps: u32,
    pixfmt: String,
}

#[derive(Clone, Debug)]
struct CameraInfo {
    uri: String,
    product: String,
    // Best RGB stream info
    resolution: Option<(u32, u32)>,
    fps: Option<u32>,
    // All available streams (for verbose mode)
    all_streams: Vec<StreamInfo>,
}

#[derive(Debug, Clone)]
enum CameraCommand {
    Refresh,
    Select(String),
    SelectWithResolution(String, u32, u32, u32), // uri, width, height, fps
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
                    "{:.2} µm/px",
                    self.result.map[self.active.unwrap()]
                ))));
            }
        });

        ui.response()
    }
}

struct StreamD {
    st: eye::hal::platform::Stream<'static>,
    d: [usize; 2],
}

fn list_cameras() -> Vec<CameraInfo> {
    let mut cameras = Vec::new();
    if let Some(ctx) = PlatformContext::all().next() {
        if let Ok(dev_descrs) = ctx.devices() {
            for dev in dev_descrs {
                let mut all_streams = Vec::new();
                let mut best_resolution = None;
                let mut best_fps = None;

                if let Ok(device) = ctx.open_device(&dev.uri) {
                    if let Ok(device) = Device::new(device) {
                        if let Ok(streams) = device.streams() {
                            // Collect all RGB streams
                            for s in streams.iter() {
                                if s.pixfmt == PixelFormat::Rgb(24) && s.width <= 2560 {
                                    let fps =
                                        (1000.0 / s.interval.as_millis() as f32).round() as u32;
                                    all_streams.push(StreamInfo {
                                        width: s.width,
                                        height: s.height,
                                        fps,
                                        pixfmt: "RGB24".to_string(),
                                    });
                                }
                            }

                            // Sort by resolution (descending), then fps (descending)
                            all_streams.sort_by(|a, b| {
                                let res_a = (a.width as u64) * (a.height as u64);
                                let res_b = (b.width as u64) * (b.height as u64);
                                res_b.cmp(&res_a).then(b.fps.cmp(&a.fps))
                            });

                            // Deduplicate (same resolution+fps)
                            all_streams.dedup_by(|a, b| {
                                a.width == b.width && a.height == b.height && a.fps == b.fps
                            });

                            // Find best stream
                            let best = streams
                                .into_iter()
                                .filter(|x| x.pixfmt == PixelFormat::Rgb(24))
                                .filter(|x| x.width <= 2560)
                                .max_by_key(|d| {
                                    d.width as u128 * d.height as u128 / d.interval.as_millis()
                                });
                            if let Some(s) = best {
                                let fps = (1000.0 / s.interval.as_millis() as f32).round() as u32;
                                best_resolution = Some((s.width, s.height));
                                best_fps = Some(fps);
                            }
                        }
                    }
                }

                cameras.push(CameraInfo {
                    uri: dev.uri.clone(),
                    product: dev.product.clone(),
                    resolution: best_resolution,
                    fps: best_fps,
                    all_streams,
                });
            }
        }
    }
    cameras
}

fn find_stream_for_camera(uri: &str) -> anyhow::Result<Option<StreamD>> {
    let ctx = if let Some(ctx) = PlatformContext::all().next() {
        ctx
    } else {
        return Ok(None);
    };

    let dev = ctx.open_device(uri)?;
    let dev = Device::new(dev)?;
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

    let stream = dev.start_stream(&stream_descr)?;

    Ok(Some(StreamD {
        st: stream,
        d: dimensions,
    }))
}

fn find_stream_for_camera_with_resolution(
    uri: &str,
    width: u32,
    height: u32,
    target_fps: u32,
) -> anyhow::Result<Option<StreamD>> {
    let ctx = if let Some(ctx) = PlatformContext::all().next() {
        ctx
    } else {
        return Ok(None);
    };

    let dev = ctx.open_device(uri)?;
    let dev = Device::new(dev)?;

    // Find stream matching the requested resolution and fps
    let matching = dev
        .streams()?
        .into_iter()
        .filter(|x| x.pixfmt == PixelFormat::Rgb(24))
        .filter(|x| x.width == width && x.height == height)
        .min_by_key(|d| {
            let fps = (1000.0 / d.interval.as_millis() as f32).round() as i32;
            (fps - target_fps as i32).abs()
        });

    let stream_descr = match matching {
        Some(s) => s,
        None => return Ok(None),
    };
    let dimensions = [stream_descr.width as usize, stream_descr.height as usize];

    let stream = dev.start_stream(&stream_descr)?;

    Ok(Some(StreamD {
        st: stream,
        d: dimensions,
    }))
}

fn find_stream() -> anyhow::Result<Option<StreamD>> {
    let ctx = if let Some(ctx) = PlatformContext::all().next() {
        ctx
    } else {
        return Ok(None);
    };

    // Create a list of valid capture devices in the system.
    let dev_descrs = ctx.devices()?;

    if dev_descrs.len() == 0 {
        return Ok(None);
    }
    // Print the supported formats for each device.
    let dev = ctx.open_device(&dev_descrs[0].uri)?;
    // let dev = ctx.open_device("v4l:///dev/video")?;
    let dev = Device::new(dev)?;
    let maxxed = dev
        .streams()?
        .into_iter()
        .filter(|x| x.pixfmt == PixelFormat::Rgb(24))
        .filter(|x| x.width <= 2560)
        .max_by_key(|d| d.width as u128 * d.height as u128 / d.interval.as_millis())
        .unwrap();

    let stream_descr = maxxed;
    let dimensions = [stream_descr.width as usize, stream_descr.height as usize];

    let stream = dev.start_stream(&stream_descr)?;

    Ok(Some(StreamD {
        st: stream,
        d: dimensions,
    }))
}

async fn camera_task(
    cameras: Arc<RwLock<Arc<Vec<CameraInfo>>>>,
    selected_uri: Arc<RwLock<Option<String>>>,
    mut camera_cmd_rx: mpsc::UnboundedReceiver<CameraCommand>,
    waiting: Arc<TokioMutex<Option<StreamD>>>,
) {
    // Initial camera list (no stream open yet, safe to enumerate)
    let initial_cameras = list_cameras();
    let first_uri = initial_cameras.first().map(|c| c.uri.clone());
    *cameras.write().await = Arc::new(initial_cameras);

    // Open first camera
    if let Some(uri) = first_uri {
        if let Ok(Some(s)) = find_stream_for_camera(&uri) {
            *waiting.lock().await = Some(s);
            *selected_uri.write().await = Some(uri);
        }
    }

    loop {
        let Some(cmd) = camera_cmd_rx.recv().await else {
            break;
        };

        match cmd {
            CameraCommand::Refresh => {
                let old_uri = selected_uri.read().await.clone();

                // Drop stream FIRST to release device
                *waiting.lock().await = None;

                // Now safe to enumerate
                let new_cameras = list_cameras();
                *cameras.write().await = Arc::new(new_cameras);

                // Re-open previous camera if it existed
                if let Some(uri) = old_uri {
                    if let Ok(Some(s)) = find_stream_for_camera(&uri) {
                        *waiting.lock().await = Some(s);
                    }
                }
            }
            CameraCommand::Select(uri) => {
                // Check if already selected
                let already_selected = selected_uri
                    .read()
                    .await
                    .as_ref()
                    .map_or(false, |current| current == &uri);

                if already_selected {
                    continue;
                }

                // Drop stream FIRST to release device
                *waiting.lock().await = None;

                // Update selected URI
                *selected_uri.write().await = Some(uri.clone());

                // Open new stream
                if let Ok(Some(s)) = find_stream_for_camera(&uri) {
                    *waiting.lock().await = Some(s);
                }
            }
            CameraCommand::SelectWithResolution(uri, width, height, fps) => {
                // Drop stream FIRST to release device
                *waiting.lock().await = None;

                // Update selected URI
                *selected_uri.write().await = Some(uri.clone());

                // Open new stream with specific resolution
                match find_stream_for_camera_with_resolution(&uri, width, height, fps) {
                    Ok(Some(s)) => {
                        *waiting.lock().await = Some(s);
                    }
                    Ok(None) => {
                        // Fallback to best stream if specific resolution not found
                        if let Ok(Some(s)) = find_stream_for_camera(&uri) {
                            *waiting.lock().await = Some(s);
                        }
                    }
                    Err(_) => {}
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async { async_main().await })
}

async fn async_main() -> Result<()> {
    let (camera_cmd_tx, camera_cmd_rx) = mpsc::unbounded_channel::<CameraCommand>();

    // Shared state for camera list
    let available_cameras: Arc<RwLock<Arc<Vec<CameraInfo>>>> =
        Arc::new(RwLock::new(Arc::new(Vec::new())));
    let selected_camera_uri: Arc<RwLock<Option<String>>> = Arc::new(RwLock::new(None));
    let waiting: Arc<TokioMutex<Option<StreamD>>> = Arc::new(TokioMutex::new(None));

    let cameras_for_task = available_cameras.clone();
    let selected_for_task = selected_camera_uri.clone();
    // Spawn camera task
    tokio::spawn(camera_task(
        cameras_for_task,
        selected_for_task,
        camera_cmd_rx,
        waiting.clone(),
    ));

    let _ = eframe::run_native(
        "UVC Camera",
        NativeOptions::default(),
        Box::new(move |ctx| {
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
                waiting,
                available_cameras,
                selected_camera_uri,
                selected_stream: None,
                camera_cmd_tx,
                verbose_camera_ui: false,
            };

            app.load_profiles()?;

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

            // Clone an Arc to avoid holding locks (and avoid cloning the Vec) during UI rendering
            let cameras: Arc<Vec<CameraInfo>> = self
                .available_cameras
                .try_read()
                .map(|c| c.clone())
                .unwrap_or_else(|_| Arc::new(Vec::new()));
            let selected_uri: Option<String> = self
                .selected_camera_uri
                .try_read()
                .ok()
                .and_then(|s| s.clone());

            if cameras.is_empty() {
                ui.label("No cameras found");
            } else {
                // Verbose mode toggle
                ui.checkbox(&mut self.verbose_camera_ui, "Show all streams");
                ui.add_space(5.);

                let mut camera_cmd: Option<CameraCommand> = None;

                if self.verbose_camera_ui {
                    // Verbose mode: show all streams grouped by device
                    for cam in cameras.iter() {
                        let is_device_selected =
                            selected_uri.as_ref().map_or(false, |u| u == &cam.uri);
                        let dev_name = cam.uri.split('/').last().unwrap_or(&cam.uri);

                        // Device header
                        let header_fill = if is_device_selected {
                            Color32::from_rgb(50, 65, 50)
                        } else {
                            Color32::from_rgb(35, 35, 40)
                        };

                        egui::Frame::new()
                            .fill(header_fill)
                            .stroke(Stroke::new(1.0, Color32::DARK_GRAY))
                            .corner_radius(4.0)
                            .inner_margin(6.0)
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(RichText::new(dev_name).strong().color(Color32::WHITE));
                                ui.label(
                                    RichText::new(&cam.product)
                                        .small()
                                        .color(Color32::LIGHT_GRAY),
                                );
                            });

                        // Stream list for this device
                        if cam.all_streams.is_empty() {
                            ui.indent("no_streams", |ui| {
                                ui.label(
                                    RichText::new("  no RGB streams")
                                        .small()
                                        .color(Color32::GRAY),
                                );
                            });
                        } else {
                            for stream in &cam.all_streams {
                                let stream_info = format!(
                                    "  {}x{} {}fps",
                                    stream.width, stream.height, stream.fps
                                );
                                let stream_id = format!(
                                    "{}_{}x{}_{}",
                                    cam.uri, stream.width, stream.height, stream.fps
                                );

                                // Check if this stream is selected
                                let is_stream_selected = is_device_selected
                                    && self.selected_stream.map_or(false, |(w, h, f)| {
                                        w == stream.width && h == stream.height && f == stream.fps
                                    });

                                // Pre-check hover state
                                let hover_rect = ui.available_rect_before_wrap();
                                let is_hovered = ui.rect_contains_pointer(
                                    hover_rect.with_max_y(hover_rect.min.y + 24.0),
                                );

                                let stream_fill = if is_stream_selected {
                                    Color32::from_rgb(60, 90, 60)
                                } else if is_hovered {
                                    Color32::from_rgb(55, 55, 65)
                                } else {
                                    Color32::from_rgb(45, 45, 50)
                                };

                                let text_color = if is_stream_selected {
                                    Color32::from_rgb(180, 255, 180)
                                } else {
                                    Color32::from_rgb(150, 200, 150)
                                };

                                let frame_resp = egui::Frame::new()
                                    .fill(stream_fill)
                                    .corner_radius(2.0)
                                    .inner_margin(4.0)
                                    .show(ui, |ui| {
                                        ui.set_width(ui.available_width());
                                        ui.label(
                                            RichText::new(&stream_info).small().color(text_color),
                                        );
                                    });

                                let click_resp = ui.interact(
                                    frame_resp.response.rect,
                                    egui::Id::new(&stream_id),
                                    Sense::click(),
                                );
                                if click_resp.clicked() {
                                    self.selected_stream =
                                        Some((stream.width, stream.height, stream.fps));
                                    camera_cmd = Some(CameraCommand::SelectWithResolution(
                                        cam.uri.clone(),
                                        stream.width,
                                        stream.height,
                                        stream.fps,
                                    ));
                                }
                                if click_resp.hovered() {
                                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                }
                            }
                        }
                        ui.add_space(6.0);
                    }
                } else {
                    // Simple mode: show only best stream per device
                    for cam in cameras.iter() {
                        let is_selected = selected_uri.as_ref().map_or(false, |u| u == &cam.uri);

                        // Device name from URI (e.g., video0)
                        let dev_name = cam.uri.split('/').last().unwrap_or(&cam.uri);

                        // Build info string
                        let info = if let (Some((w, h)), Some(fps)) = (cam.resolution, cam.fps) {
                            format!("{}x{} {}fps", w, h, fps)
                        } else {
                            "no RGB stream".to_string()
                        };

                        let fill = if is_selected {
                            Color32::from_rgb(60, 80, 60)
                        } else {
                            Color32::from_rgb(40, 40, 45)
                        };

                        let frame_resp = egui::Frame::new()
                            .fill(fill)
                            .stroke(Stroke::new(1.0, Color32::GRAY))
                            .corner_radius(4.0)
                            .inner_margin(6.0)
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.vertical(|ui| {
                                    ui.label(
                                        RichText::new(dev_name).strong().color(Color32::WHITE),
                                    );
                                    ui.label(
                                        RichText::new(&cam.product)
                                            .small()
                                            .color(Color32::LIGHT_GRAY),
                                    );
                                    ui.label(
                                        RichText::new(&info)
                                            .small()
                                            .color(Color32::from_rgb(150, 200, 150)),
                                    );
                                });
                            });

                        let click_resp = ui.interact(
                            frame_resp.response.rect,
                            egui::Id::new(&cam.uri),
                            Sense::click(),
                        );
                        if click_resp.clicked() {
                            self.selected_stream = None; // Clear specific stream selection in simple mode
                            camera_cmd = Some(CameraCommand::Select(cam.uri.clone()));
                        }
                        if click_resp.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }

                        ui.add_space(4.0);
                    }
                }

                if let Some(cmd) = camera_cmd {
                    let _ = self.camera_cmd_tx.send(cmd);
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
            // If the lock is contended (camera task swapping streams), assume a stream exists.
            let mut is_waiting = false;
            if let Ok(mut guard) = self.waiting.try_lock() {
                if let Some(ref mut stream) = *guard {
                    // NOTE: this may block depending on backend; user requested no frame channel.
                    let buf: Option<std::result::Result<&[u8], _>> = stream.st.next();
                    if let Some(Ok(buf)) = buf {
                        let dimensions = stream.d;
                        let expected_size = dimensions[0] * dimensions[1] * 3;
                        if buf.len() == expected_size {
                            self.texture.set(
                                ColorImage::from_rgb(dimensions, buf),
                                TextureOptions::default(),
                            );
                            ctx.request_repaint();
                        }
                    } else {
                        // Stream ended; drop it.
                        *guard = None;
                        is_waiting = true;
                    }
                } else {
                    is_waiting = true;
                }
            }

            // Keep the UI ticking while a stream is open
            if !is_waiting {
                ctx.request_repaint_after(Duration::from_millis(16));
            }

            if is_waiting {
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
                    let _ = self.dump_profile();
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
