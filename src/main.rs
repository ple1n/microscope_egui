#![allow(static_mut_refs)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, mpsc};
use std::thread;

use eframe::{App, Frame, NativeOptions};
use egui::load::SizedTexture;
use egui::{
    Button, CentralPanel, Color32, ColorImage, DragValue, Grid, Id, Image, Label, LayerId, Margin,
    Painter, Pos2, Rect, RichText, Sense, Slider, Stroke, TextEdit, TextureHandle, TextureOptions,
    Ui, Widget, Window, emath, epaint, pos2,
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
            Self::X40 => "40x 0.4mm",
            Self::X100 => "100X 0.1mm",
        }
    }
    pub fn default_cal_len(&self) -> f64 {
        match &self {
            Self::X4 => 1e3,
            Self::X10 => 0.7e3,
            Self::X40 => 0.4e3,
            Self::X100 => 0.1e3,
        }
    }
}

impl Widget for &mut Calibration {
    fn ui(self, ui: &mut Ui) -> egui::Response {
        ui.vertical(|ui| {
            ui.add_space(40.);
            ui.add(Label::new(
                RichText::new("calibration").color(Color32::WHITE.gamma_multiply(0.9)),
            ));
            ui.add_space(10.);

            let result = self.result;
            for (target, val) in result.map.iter() {
                let mut btn = Button::new(target.button_text());

                if let Some(active) = self.active {
                    if target == active {
                        btn = btn.fill(Color32::from_rgb(122, 104, 1));
                    }
                } else if *val != 0. {
                    btn = btn.fill(Color32::DARK_GREEN.gamma_multiply(0.8));
                } else {
                    // grey, default
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

static mut PROFILES: Vec<std::path::PathBuf> = vec![];

fn main() -> Result<()> {
    // scan for profiles
    let profile_path= "./profiles";
    let rd = std::fs::read_dir(profile_path);
    if rd.is_err() {
        std::fs::create_dir(profile_path)?;
    } else {
        let rd = rd?;
        for entry in rd {
            let ent = entry?;
            unsafe {
                PROFILES.push(ent.path());
            }
        }
    }


    let ctx = if let Some(ctx) = PlatformContext::all().next() {
        ctx
    } else {
        return Ok(());
    };

    // Create a list of valid capture devices in the system.
    let dev_descrs = ctx.devices()?;
    // Print the supported formats for each device.
    let dev = ctx.open_device(&dev_descrs[0].uri)?;
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

    let mut stream = dev.start_stream(&stream_descr)?;

    let _ = eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|ctx| {
            let app = UVCPlayer {
                texture: ctx.egui_ctx.load_texture(
                    "vid",
                    ColorImage::example(),
                    Default::default(),
                ),
                rect_begin: None,
                rect_motion: None,
                rects: Default::default(),
                ratio: Default::default(),
                profiles: Default::default()
            };
            let mut txt = app.texture.clone();
            let ctx = ctx.egui_ctx.clone();

            thread::spawn(move || {
                loop {
                    let buf = stream.next().unwrap().unwrap();
                    txt.set(
                        ColorImage::from_rgb(dimensions, &buf),
                        TextureOptions::default(),
                    );
                    ctx.request_repaint();
                }
            });

            Ok(Box::new(app))
        }),
    );

    Ok(())
}

impl App for UVCPlayer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::SidePanel::new(egui::panel::Side::Right, "rpanel").show(ctx, |ui| {
            self.ratio.ui(ui);
        });
        CentralPanel::default().show(ctx, |ui| {
            let response = ui.add(Image::new(SizedTexture::from_handle(&self.texture)));
            let pt = ui.painter();
            let sense = response.interact(Sense::all());
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
                }
            }
            if sense.drag_stopped() {
                self.ratio.calibrating = false;
            }

            let all_rects = self.rects.iter().chain(&appeneded);
            for rect in all_rects.clone() {
                let inside_width = 5.;
                pt.add(epaint::RectShape::new(
                    *rect,
                    0,
                    Color32::TRANSPARENT,
                    Stroke::new(inside_width, Color32::BLACK.gamma_multiply(0.5)),
                    egui::StrokeKind::Outside,
                ));
                let out_rect = rect.expand(inside_width);
                pt.add(epaint::RectShape::new(
                    out_rect,
                    0,
                    Color32::TRANSPARENT,
                    Stroke::new(5., Color32::WHITE.gamma_multiply(0.5)),
                    egui::StrokeKind::Outside,
                ));
            }

            for rect in all_rects {
                let rect_lb_x = Rect::EVERYTHING
                    .with_min_x(rect.left() - 100.)
                    .with_max_x(rect.right() + 100.)
                    .with_min_y(rect.top() - 60.)
                    .with_max_y(rect.top() - 8.);
                let wd = format!(" {}px ", rect.width().round());
                let lb = RichText::new(wd)
                    .color(Color32::WHITE)
                    .size(40.)
                    .background_color(Color32::BLACK.gamma_multiply(0.5));
                let lb = Label::new(lb);
                let rect_lb_y = Rect::EVERYTHING
                    .with_min_x(rect.right() - 30.)
                    .with_min_y(rect.top())
                    .with_max_y(rect.bottom())
                    .with_max_x(rect.right() + 180.);

                ui.put(rect_lb_x, lb);
                let ht = format!(" {}px ", rect.height().round());
                let lb = RichText::new(ht)
                    .color(Color32::WHITE)
                    .size(40.)
                    .background_color(Color32::BLACK.gamma_multiply(0.5));
                let lb = Label::new(lb);

                ui.put(rect_lb_y, lb);
            }

            response
        });
    }
}
