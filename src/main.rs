#![allow(static_mut_refs)]

mod dmabuf;

use core::f32;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::str::FromStr;
use std::{fs, time::Duration};

use parking_lot::RwLock as ParkingRwLock;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{RwLock, mpsc, watch};

use eframe::{App, NativeOptions};
use egui::epaint::PathStroke;
use egui::load::SizedTexture;
use egui::{
    Button, CentralPanel, Color32, Image, Label, Pos2, Rect, RichText, Sense, Stroke, TextEdit,
    TextWrapMode, Ui, Widget, epaint, pos2,
};

use enum_map::{Enum, EnumMap};
use eye::colorconvert::Device;
use eye::hal::PlatformContext;
use eye::hal::format::PixelFormat;
#[cfg(target_os = "linux")]
use eye::hal::platform::v4l2::dma::Stream as DmaStream;
use eye::hal::traits::{Context as _, Device as _, Stream as _};

use anyhow::Result;
use anyhow::bail;
use image::RgbaImage;
use serde::{Deserialize, Serialize};
#[cfg(target_os = "linux")]
use std::os::fd::OwnedFd;
use std::time::{SystemTime, UNIX_EPOCH};

struct UVCPlayer {
    render_state: eframe::egui_wgpu::RenderState,
    dmabuf_status: String,
    video_texture: eframe::wgpu::Texture,
    video_texture_view: eframe::wgpu::TextureView,
    video_texture_id: egui::TextureId,
    video_size: [u32; 2],
    rgb_buffer: eframe::wgpu::Buffer,
    rgb_buffer_capacity: u64,
    rgb_dimensions: eframe::wgpu::Buffer,
    inverse_rect_buffer: eframe::wgpu::Buffer,
    inverse_rect_capacity: u64,
    gpu_rects: Vec<[f32; 8]>,
    rgb_bind_group_layout: eframe::wgpu::BindGroupLayout,
    rgb_pipeline: eframe::wgpu::ComputePipeline,
    rgb_bind_group: Option<eframe::wgpu::BindGroup>,
    yuyv_bind_group_layout: eframe::wgpu::BindGroupLayout,
    yuyv_pipeline: eframe::wgpu::ComputePipeline,
    yuyv_bind_group: Option<eframe::wgpu::BindGroup>,
    #[cfg(target_os = "linux")]
    yuyv_texture: Option<eframe::wgpu::Texture>,
    #[cfg(target_os = "linux")]
    yuyv_texture_view: Option<eframe::wgpu::TextureView>,
    #[cfg(target_os = "linux")]
    dma_in_flight: Option<u32>,
    shared_frame: Arc<ParkingRwLock<SharedFrame>>,
    shutdown_tx: watch::Sender<bool>,
    last_frame_seq: u64,

    rects: Vec<SelectionRect>,
    rect_begin: Option<Pos2>,

    rect_motion: Option<Pos2>,
    rotation_angle: f32,
    rotation_calibrating: bool,
    rotation_begin: Option<Pos2>,
    rotation_motion: Option<Pos2>,
    ratio: Calibration,
    profiles: BTreeMap<PathBuf, MicroscopeRatio>,
    active_profile: Option<PathBuf>,
    new_profile_name: String,

    show_labels: bool,

    // Camera selection
    available_cameras: Arc<RwLock<Arc<Vec<CameraInfo>>>>,
    selected_camera_uri: Arc<RwLock<Option<String>>>,
    selected_stream: Option<(u32, u32, u32)>, // (width, height, fps)
    camera_cmd_tx: mpsc::UnboundedSender<CameraCommand>,
    verbose_camera_ui: bool,

    // Capture UI/state
    save_path: String,
    last_capture_status: Option<String>,
}

#[derive(Clone, Copy)]
struct SelectionRect {
    corners: [Pos2; 4],
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

#[derive(Clone, Debug)]
struct StreamSelection {
    uri: String,
    resolution: Option<(u32, u32, u32)>,
}

#[derive(Debug, Default)]
struct SharedFrame {
    seq: u64,
    width: u32,
    height: u32,
    /// Native RGB24 bytes. Keeping this packed avoids a per-pixel CPU expansion.
    rgb: Vec<u8>,
    #[cfg(target_os = "linux")]
    dma: Option<(OwnedFd, dmabuf::DmaBufFrame)>,
    #[cfg(target_os = "linux")]
    dma_requeue: Option<u32>,
}

#[derive(Default, Clone, Copy, Serialize)]
struct MicroscopeRatio {
    /// Microns per pixel
    map: EnumMap<Magnification, f64>,
    rotation_angle: f32,
}

#[derive(Deserialize)]
struct MicroscopeRatioData {
    #[serde(default)]
    map: HashMap<String, f64>,
    #[serde(default)]
    rotation_angle: f32,
}

impl<'de> Deserialize<'de> for MicroscopeRatio {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = MicroscopeRatioData::deserialize(deserializer)?;
        let mut ratio = Self::default();
        let rotation_angle = data.rotation_angle;

        for (name, value) in data.map {
            let target = match name.as_str() {
                "X4" => Magnification::X4,
                "X10" => Magnification::X10,
                "X40" => Magnification::X40,
                "X100" => Magnification::X100,
                "Mm4" => Magnification::Mm4,
                _ => {
                    return Err(serde::de::Error::custom(format!(
                        "unknown calibration key {name}"
                    )));
                }
            };
            ratio.map[target] = value;
        }
        ratio.rotation_angle = rotation_angle;

        Ok(ratio)
    }
}

#[derive(Default, Clone, Copy)]
struct Calibration {
    result: MicroscopeRatio,
    active: Option<Magnification>,
    calibrating: bool,
    calibrate_by: CalibrationAxis,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum CalibrationAxis {
    #[default]
    Width,
    Height,
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
    Mm4,
}

impl Magnification {
    pub fn button_text(self) -> &'static str {
        match self {
            Self::X4 => "4x 1mm",
            Self::X10 => "10x 0.7mm",
            Self::X40 => "40x 0.15mm",
            Self::X100 => "100X 0.05mm",
            Self::Mm4 => "4mm",
        }
    }
    pub fn default_cal_len(&self) -> f64 {
        match &self {
            Self::X4 => 1e3,
            Self::X10 => 0.7e3,
            Self::X40 => 0.15e3,
            Self::X100 => 0.05e3,
            Self::Mm4 => 4e3,
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

            for (group_name, targets) in [
                (
                    "Microscope",
                    [
                        Magnification::X4,
                        Magnification::X10,
                        Magnification::X40,
                        Magnification::X100,
                    ]
                    .as_slice(),
                ),
                ("Camera", [Magnification::Mm4].as_slice()),
            ] {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.label(RichText::new(group_name).strong().color(Color32::WHITE));
                    ui.add_space(4.);

                    for target in targets {
                        let val = self.result.map[*target];
                        let mut btn = Button::new(target.button_text());

                        if val > 0. {
                            btn = btn.fill(Color32::DARK_GREEN.gamma_multiply(0.8));
                        }

                        if self.active == Some(*target) {
                            btn = btn.fill(Color32::from_rgb(122, 104, 1));
                        }

                        if ui.add(btn).clicked() {
                            self.calibrating = false;
                            self.active = Some(*target);
                        }
                    }
                });
                ui.add_space(6.);
            }

            if self.active.is_some() {
                ui.add_space(20.);
                ui.label(
                    RichText::new("Calibration by")
                        .strong()
                        .color(Color32::WHITE),
                );
                ui.horizontal(|ui| {
                    let width_button =
                        ui.add(Button::new("Width").selected(
                            self.calibrating && self.calibrate_by == CalibrationAxis::Width,
                        ));
                    let height_button = ui.add(Button::new("Height").selected(
                        self.calibrating && self.calibrate_by == CalibrationAxis::Height,
                    ));

                    if width_button.clicked() {
                        if self.calibrating && self.calibrate_by == CalibrationAxis::Width {
                            self.calibrating = false;
                        } else {
                            self.calibrate_by = CalibrationAxis::Width;
                            self.calibrating = true;
                        }
                    } else if height_button.clicked() {
                        if self.calibrating && self.calibrate_by == CalibrationAxis::Height {
                            self.calibrating = false;
                        } else {
                            self.calibrate_by = CalibrationAxis::Height;
                            self.calibrating = true;
                        }
                    }
                });

                if self.calibrating {
                    let active = self.active.unwrap();
                    ui.label(format!(
                        "Drag a rectangle over a known {} target using its {}.",
                        active.button_text(),
                        match self.calibrate_by {
                            CalibrationAxis::Width => "width",
                            CalibrationAxis::Height => "height",
                        }
                    ));
                    ui.label("The selected dimension will be tagged in the image.");
                } else {
                    ui.label(format!(
                        "{:.2} µm/px",
                        self.result.map[self.active.unwrap()]
                    ));
                }
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
    stream_sel_tx: watch::Sender<Option<StreamSelection>>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    // Initial camera list (no stream open yet, safe to enumerate)
    let initial_cameras = list_cameras();
    let first_uri = initial_cameras.first().map(|c| c.uri.clone());
    *cameras.write().await = Arc::new(initial_cameras);

    // Select first camera
    if let Some(uri) = first_uri {
        *selected_uri.write().await = Some(uri.clone());
        let _ = stream_sel_tx.send_replace(Some(StreamSelection {
            uri,
            resolution: None,
        }));
    }

    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    break;
                }
            }
            maybe_cmd = camera_cmd_rx.recv() => {
                let Some(cmd) = maybe_cmd else {
                    break;
                };

                match cmd {
                    CameraCommand::Refresh => {
                let old_uri = selected_uri.read().await.clone();

                // Now safe to enumerate
                let new_cameras = list_cameras();
                *cameras.write().await = Arc::new(new_cameras);

                // Re-open previous camera if it existed
                if let Some(uri) = old_uri {
                    let _ = stream_sel_tx.send_replace(Some(StreamSelection {
                        uri,
                        resolution: None,
                    }));
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

                // Update selected URI
                *selected_uri.write().await = Some(uri.clone());

                let _ = stream_sel_tx.send_replace(Some(StreamSelection {
                    uri,
                    resolution: None,
                }));
                    }
                    CameraCommand::SelectWithResolution(uri, width, height, fps) => {
                // Update selected URI
                *selected_uri.write().await = Some(uri.clone());

                let _ = stream_sel_tx.send_replace(Some(StreamSelection {
                    uri,
                    resolution: Some((width, height, fps)),
                }));
                    }
                }
            }
        }
    }
}

async fn frame_pump_task(
    mut stream_sel_rx: watch::Receiver<Option<StreamSelection>>,
    shared_frame: Arc<ParkingRwLock<SharedFrame>>,
    repaint_context: Arc<ParkingRwLock<Option<egui::Context>>>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut current_stop: Option<Arc<AtomicBool>> = None;
    let mut current_handle: Option<tokio::task::JoinHandle<()>> = None;

    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    break;
                }
            }
            changed = stream_sel_rx.changed() => {
                if changed.is_err() {
                    break;
                }
            }
        }

        if let Some(stop) = current_stop.take() {
            stop.store(true, Ordering::Relaxed);
        }
        if let Some(handle) = current_handle.take() {
            // Don't block the async runtime forever if the camera read is stuck.
            let _ = tokio::time::timeout(Duration::from_millis(200), handle).await;
        }

        let Some(sel) = stream_sel_rx.borrow().clone() else {
            let mut frame = shared_frame.write();
            *frame = SharedFrame::default();
            continue;
        };

        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_task = stop.clone();
        current_stop = Some(stop);

        let shared_frame = shared_frame.clone();
        let repaint_context = repaint_context.clone();
        current_handle = Some(tokio::task::spawn_blocking(move || {
            #[cfg(target_os = "linux")]
            if let Some(path) = sel.uri.strip_prefix("v4l://") {
                let dma_result = match sel.resolution {
                    Some((width, height, fps)) => {
                        eprintln!(
                            "capture: attempting V4L2 DMA-BUF path for {path} \
                                 ({width}x{height}@{fps}) using YUYV"
                        );
                        DmaStream::open(path, width, height, fps)
                    }
                    None => {
                        eprintln!(
                            "capture: attempting V4L2 DMA-BUF path for {path} \
                                 (auto mode) using YUYV"
                        );
                        DmaStream::open_best(path)
                    }
                };
                match dma_result {
                    Ok(mut dma_stream) => {
                        eprintln!("capture: V4L2 DMA-BUF path active for {path}");
                        let mut seq = 0u64;
                        while !stop_for_task.load(Ordering::Relaxed) {
                            if let Some(index) = shared_frame.write().dma_requeue.take() {
                                if let Err(error) = dma_stream.requeue(index) {
                                    eprintln!(
                                        "capture: DMA-BUF requeue failed for buffer {index}: {error}"
                                    );
                                    break;
                                }
                            }
                            let dma_frame = match dma_stream.next() {
                                Ok(frame) => frame,
                                Err(error) => {
                                    eprintln!("capture: DMA-BUF dequeue failed: {error}");
                                    break;
                                }
                            };
                            let metadata = dmabuf::DmaBufFrame {
                                buffer_index: dma_frame.metadata.buffer_index,
                                width: dma_frame.metadata.width,
                                height: dma_frame.metadata.height,
                                fourcc: dma_frame.metadata.fourcc,
                                modifier: 0,
                                stride: dma_frame.metadata.stride,
                                offset: dma_frame.metadata.offset,
                            };
                            let mut frame = shared_frame.write();
                            frame.width = metadata.width;
                            frame.height = metadata.height;
                            frame.dma = Some((dma_frame.fd, metadata));
                            seq = seq.wrapping_add(1);
                            frame.seq = seq;
                            if let Some(ctx) = repaint_context.read().clone() {
                                ctx.request_repaint();
                            }
                        }
                        return;
                    }
                    Err(error) => {
                        eprintln!(
                            "capture: DMA-BUF unavailable for {path}: {error}; \
                             falling back to CPU capture"
                        );
                    }
                }
            }

            let open = match sel.resolution {
                Some((w, h, fps)) => find_stream_for_camera_with_resolution(&sel.uri, w, h, fps)
                    .ok()
                    .flatten(),
                None => find_stream_for_camera(&sel.uri).ok().flatten(),
            };

            let Some(mut stream) = open else {
                eprintln!("capture: CPU fallback could not open {}", sel.uri);
                let mut frame = shared_frame.write();
                *frame = SharedFrame::default();
                return;
            };

            let width = stream.d[0] as u32;
            let height = stream.d[1] as u32;
            eprintln!(
                "capture: CPU RGB24 fallback active for {} ({}x{})",
                sel.uri, width, height
            );
            // One shared packed RGB buffer; the UI reads with try_read, so it won't block.
            {
                let mut frame = shared_frame.write();
                frame.width = width;
                frame.height = height;
                let needed_len = (width as usize) * (height as usize) * 3;
                if frame.rgb.len() != needed_len {
                    frame.rgb.resize(needed_len, 0);
                }
            }

            let mut seq: u64 = 0;
            while !stop_for_task.load(Ordering::Relaxed) {
                let buf: Option<std::result::Result<&[u8], _>> = stream.st.next();
                let Some(Ok(rgb)) = buf else {
                    break;
                };

                if rgb.len() != (width as usize * height as usize * 3) {
                    continue;
                }

                {
                    // Try to keep UI responsive: if UI is reading, skip this frame.
                    let mut frame = shared_frame.write();

                    // Upload packed RGB24 and expand to RGBA in the GPU compute pass.
                    frame.rgb.copy_from_slice(rgb);

                    seq = seq.wrapping_add(1);
                    frame.seq = seq;
                }

                if let Some(ctx) = repaint_context.read().clone() {
                    ctx.request_repaint();
                }
            }
        }));
    }

    // Shutdown/close requested: signal the blocking capture loop to stop.
    if let Some(stop) = current_stop.take() {
        stop.store(true, Ordering::Relaxed);
    }
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async { async_main().await })
}

fn native_options() -> NativeOptions {
    let mut options = NativeOptions::default();
    let requested_gpu = std::env::var("VID_UVC_GPU").ok();
    let vulkan_only = std::env::var("VID_UVC_VULKAN_ONLY")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        if vulkan_only {
            setup.instance_descriptor.backends = eframe::wgpu::Backends::VULKAN;
        }

        setup.native_adapter_selector = Some(std::sync::Arc::new(move |adapters, _surface| {
            if let Some(requested_gpu) = requested_gpu.as_deref() {
                return adapters
                    .iter()
                    .find(|adapter| {
                        adapter
                            .get_info()
                            .name
                            .to_ascii_lowercase()
                            .contains(&requested_gpu.to_ascii_lowercase())
                    })
                    .cloned()
                    .ok_or_else(|| {
                        format!("VID_UVC_GPU did not match an adapter: {requested_gpu}")
                    });
            }

            adapters
                .iter()
                .find(|adapter| {
                    adapter.get_info().device_type == eframe::wgpu::DeviceType::IntegratedGpu
                })
                .or_else(|| adapters.first())
                .cloned()
                .ok_or_else(|| "wgpu did not expose any adapters".to_owned())
        }));
    }

    options
}

async fn async_main() -> Result<()> {
    let (camera_cmd_tx, camera_cmd_rx) = mpsc::unbounded_channel::<CameraCommand>();
    let (stream_sel_tx, stream_sel_rx) = watch::channel::<Option<StreamSelection>>(None);
    let (shutdown_tx, shutdown_rx) = watch::channel::<bool>(false);

    let shared_frame: Arc<ParkingRwLock<SharedFrame>> =
        Arc::new(ParkingRwLock::new(SharedFrame::default()));
    let repaint_context: Arc<ParkingRwLock<Option<egui::Context>>> =
        Arc::new(ParkingRwLock::new(None));

    // Shared state for camera list
    let available_cameras: Arc<RwLock<Arc<Vec<CameraInfo>>>> =
        Arc::new(RwLock::new(Arc::new(Vec::new())));
    let selected_camera_uri: Arc<RwLock<Option<String>>> = Arc::new(RwLock::new(None));

    let cameras_for_task = available_cameras.clone();
    let selected_for_task = selected_camera_uri.clone();

    tokio::spawn(frame_pump_task(
        stream_sel_rx,
        shared_frame.clone(),
        repaint_context.clone(),
        shutdown_rx.clone(),
    ));

    // Spawn camera task
    tokio::spawn(camera_task(
        cameras_for_task,
        selected_for_task,
        camera_cmd_rx,
        stream_sel_tx,
        shutdown_rx,
    ));

    eframe::run_native(
        "UVC Camera",
        native_options(),
        Box::new(move |ctx| {
            let render_state = ctx
                .wgpu_render_state
                .clone()
                .expect("eframe is not running with the wgpu renderer");
            *repaint_context.write() = Some(ctx.egui_ctx.clone());
            let dmabuf_status = dmabuf::support_status(&render_state.adapter);
            if dmabuf::is_supported(&render_state.adapter) {
                eprintln!("{dmabuf_status}; DMA-BUF importer is available");
            } else {
                eprintln!("{dmabuf_status}; using GPU RGB upload/conversion fallback");
            }

            let device = &render_state.device;

            let initial_w = 256u32;
            let initial_h = 256u32;
            let video_texture = device.create_texture(&eframe::wgpu::TextureDescriptor {
                label: Some("video_texture"),
                size: eframe::wgpu::Extent3d {
                    width: initial_w,
                    height: initial_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: eframe::wgpu::TextureDimension::D2,
                format: eframe::wgpu::TextureFormat::Rgba8Unorm,
                usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING
                    | eframe::wgpu::TextureUsages::STORAGE_BINDING
                    | eframe::wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let video_texture_view = video_texture.create_view(&Default::default());

            // The camera delivers packed RGB24. A storage-buffer -> storage-texture
            // compute pass performs the only RGB24 -> RGBA expansion on the GPU.
            let rgb_shader = device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor {
                label: Some("rgb24_to_rgba_compute"),
                source: eframe::wgpu::ShaderSource::Wgsl(
                    r#"
struct Dimensions { width: u32, height: u32, rect_count: u32 };
@group(0) @binding(0) var<storage, read> rgb: array<u32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;
@group(0) @binding(3) var<storage, read> inverse_rects: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= dimensions.width || id.y >= dimensions.height) { return; }
    let pixel = id.y * dimensions.width + id.x;
    let byte = pixel * 3u;
    // The unaligned 3-byte pixel may straddle words; load its bytes explicitly.
    let p0 = (rgb[byte / 4u] >> ((byte % 4u) * 8u)) & 0xffu;
    let p1 = (rgb[(byte + 1u) / 4u] >> (((byte + 1u) % 4u) * 8u)) & 0xffu;
    let p2 = (rgb[(byte + 2u) / 4u] >> (((byte + 2u) % 4u) * 8u)) & 0xffu;
    var color = vec3<f32>(f32(p0), f32(p1), f32(p2)) / 255.0;
    for (var i = 0u; i < dimensions.rect_count; i++) {
        let axis_x = inverse_rects[i * 2u];
        let axis_y = inverse_rects[i * 2u + 1u];
        let rel = vec2<f32>(f32(id.x), f32(id.y)) - axis_x.xy;
        let x = dot(rel, axis_x.zw);
        let y = dot(rel, axis_y.xy);
        let width = axis_y.z;
        let height = axis_y.w;
        let border = 3.0;
        let edge_distance = min(min(x, width - x), min(y, height - y));
        if (edge_distance >= 0.0 && edge_distance < border) {
            let outer_coverage = smoothstep(0.0, 1.0, edge_distance);
            let inner_coverage = 1.0 - smoothstep(border - 1.0, border, edge_distance);
            let coverage = outer_coverage * inner_coverage;
            color = mix(color, vec3<f32>(1.0) - color, coverage);
        }
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
"#
                    .into(),
                ),
            });
            let rgb_bind_group_layout =
                device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
                    label: Some("rgb24_compute_bind_group_layout"),
                    entries: &[
                        eframe::wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: eframe::wgpu::ShaderStages::COMPUTE,
                            ty: eframe::wgpu::BindingType::Buffer {
                                ty: eframe::wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        eframe::wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: eframe::wgpu::ShaderStages::COMPUTE,
                            ty: eframe::wgpu::BindingType::StorageTexture {
                                access: eframe::wgpu::StorageTextureAccess::WriteOnly,
                                format: eframe::wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: eframe::wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        eframe::wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: eframe::wgpu::ShaderStages::COMPUTE,
                            ty: eframe::wgpu::BindingType::Buffer {
                                ty: eframe::wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        eframe::wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: eframe::wgpu::ShaderStages::COMPUTE,
                            ty: eframe::wgpu::BindingType::Buffer {
                                ty: eframe::wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let rgb_pipeline =
                device.create_compute_pipeline(&eframe::wgpu::ComputePipelineDescriptor {
                    label: Some("rgb24_to_rgba_compute_pipeline"),
                    layout: Some(&device.create_pipeline_layout(
                        &eframe::wgpu::PipelineLayoutDescriptor {
                            label: Some("rgb24_compute_pipeline_layout"),
                            bind_group_layouts: &[Some(&rgb_bind_group_layout)],
                            immediate_size: 0,
                        },
                    )),
                    module: &rgb_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let yuyv_shader = device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor {
                label: Some("yuyv422_to_rgba_compute"),
                source: eframe::wgpu::ShaderSource::Wgsl(
                    r#"
struct Dimensions { width: u32, height: u32, rect_count: u32 };
@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;
@group(0) @binding(3) var<storage, read> inverse_rects: array<vec4<f32>>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= dimensions.width || id.y >= dimensions.height) { return; }
    let pair = (id.x / 2u) * 4u;
    let y_offset = select(0u, 2u, (id.x & 1u) != 0u);
    let y = textureLoad(input, vec2<i32>(i32(pair + y_offset), i32(id.y)), 0).r * 255.0;
    let u = textureLoad(input, vec2<i32>(i32(pair + 1u), i32(id.y)), 0).r * 255.0;
    let v = textureLoad(input, vec2<i32>(i32(pair + 3u), i32(id.y)), 0).r * 255.0;
    let yf = max(y - 16.0, 0.0);
    let uf = u - 128.0;
    let vf = v - 128.0;
    let rgb = vec3<f32>(1.164383 * yf + 1.596027 * vf,
        1.164383 * yf - 0.391762 * uf - 0.812968 * vf,
        1.164383 * yf + 2.017232 * uf) / 255.0;
    var color = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    for (var i = 0u; i < dimensions.rect_count; i++) {
        let axis_x = inverse_rects[i * 2u];
        let axis_y = inverse_rects[i * 2u + 1u];
        let rel = vec2<f32>(f32(id.x), f32(id.y)) - axis_x.xy;
        let x = dot(rel, axis_x.zw);
        let y = dot(rel, axis_y.xy);
        let width = axis_y.z;
        let height = axis_y.w;
        let border = 3.0;
        let edge_distance = min(min(x, width - x), min(y, height - y));
        if (edge_distance >= 0.0 && edge_distance < border) {
            let outer_coverage = smoothstep(0.0, 1.0, edge_distance);
            let inner_coverage = 1.0 - smoothstep(border - 1.0, border, edge_distance);
            let coverage = outer_coverage * inner_coverage;
            color = mix(color, vec3<f32>(1.0) - color, coverage);
        }
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
"#
                    .into(),
                ),
            });
            let yuyv_bind_group_layout =
                device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
                    label: Some("yuyv_compute_bind_group_layout"),
                    entries: &[
                        eframe::wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: eframe::wgpu::ShaderStages::COMPUTE,
                            ty: eframe::wgpu::BindingType::Texture {
                                sample_type: eframe::wgpu::TextureSampleType::Float {
                                    filterable: false,
                                },
                                view_dimension: eframe::wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        eframe::wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: eframe::wgpu::ShaderStages::COMPUTE,
                            ty: eframe::wgpu::BindingType::StorageTexture {
                                access: eframe::wgpu::StorageTextureAccess::WriteOnly,
                                format: eframe::wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: eframe::wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        eframe::wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: eframe::wgpu::ShaderStages::COMPUTE,
                            ty: eframe::wgpu::BindingType::Buffer {
                                ty: eframe::wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        eframe::wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: eframe::wgpu::ShaderStages::COMPUTE,
                            ty: eframe::wgpu::BindingType::Buffer {
                                ty: eframe::wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let yuyv_pipeline =
                device.create_compute_pipeline(&eframe::wgpu::ComputePipelineDescriptor {
                    label: Some("yuyv422_to_rgba_pipeline"),
                    layout: Some(&device.create_pipeline_layout(
                        &eframe::wgpu::PipelineLayoutDescriptor {
                            label: Some("yuyv_compute_pipeline_layout"),
                            bind_group_layouts: &[Some(&yuyv_bind_group_layout)],
                            immediate_size: 0,
                        },
                    )),
                    module: &yuyv_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let initial_capacity = (initial_w * initial_h * 3) as u64;
            let rgb_buffer = device.create_buffer(&eframe::wgpu::BufferDescriptor {
                label: Some("packed_rgb24"),
                size: initial_capacity.next_multiple_of(4),
                usage: eframe::wgpu::BufferUsages::STORAGE | eframe::wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let rgb_dimensions = device.create_buffer(&eframe::wgpu::BufferDescriptor {
                label: Some("rgb_dimensions"),
                size: 16,
                usage: eframe::wgpu::BufferUsages::UNIFORM | eframe::wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let inverse_rect_capacity = 256u64 * 32;
            let inverse_rect_buffer = device.create_buffer(&eframe::wgpu::BufferDescriptor {
                label: Some("inverse_rects"),
                size: inverse_rect_capacity,
                usage: eframe::wgpu::BufferUsages::STORAGE | eframe::wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let video_texture_id = {
                let mut renderer = render_state.renderer.write();
                renderer.register_native_texture(
                    device,
                    &video_texture_view,
                    eframe::wgpu::FilterMode::Linear,
                )
            };

            let mut app = UVCPlayer {
                render_state,
                dmabuf_status,
                video_texture,
                video_texture_view,
                video_texture_id,
                video_size: [initial_w, initial_h],
                rgb_buffer,
                rgb_buffer_capacity: initial_capacity.next_multiple_of(4),
                rgb_dimensions,
                inverse_rect_buffer,
                inverse_rect_capacity,
                gpu_rects: Vec::new(),
                rgb_bind_group_layout,
                rgb_pipeline,
                rgb_bind_group: None,
                yuyv_bind_group_layout,
                yuyv_pipeline,
                yuyv_bind_group: None,
                #[cfg(target_os = "linux")]
                yuyv_texture: None,
                #[cfg(target_os = "linux")]
                yuyv_texture_view: None,
                #[cfg(target_os = "linux")]
                dma_in_flight: None,
                shared_frame: shared_frame.clone(),
                shutdown_tx: shutdown_tx.clone(),
                last_frame_seq: 0,
                rect_begin: None,
                rect_motion: None,
                rects: Default::default(),
                rotation_angle: 0.0,
                rotation_calibrating: false,
                rotation_begin: None,
                rotation_motion: None,
                ratio: Default::default(),
                profiles: Default::default(),
                active_profile: None,
                new_profile_name: "0.5x".to_owned(),
                show_labels: true,
                available_cameras,
                selected_camera_uri,
                selected_stream: None,
                camera_cmd_tx,
                verbose_camera_ui: false,
                save_path: "./captures".to_owned(),
                last_capture_status: None,
            };

            app.load_profiles()?;

            Result::Ok(Box::new(app))
        }),
    )
    .map_err(|error| anyhow::anyhow!("eframe exited with an error: {error}"))?;

    Ok(())
}

impl UVCPlayer {
    const PROFILE_DIR: &str = "./profiles";

    fn dimensions_bytes(&self, width: u32, height: u32) -> Vec<u8> {
        [
            width.to_ne_bytes(),
            height.to_ne_bytes(),
            (self.gpu_rects.len() as u32).to_ne_bytes(),
            0u32.to_ne_bytes(),
        ]
        .concat()
    }

    fn rotate_screen_point(&self, point: Pos2, image_rect: Rect, angle: f32) -> Pos2 {
        let center = image_rect.center();
        let delta = point - center;
        let (sin, cos) = angle.sin_cos();
        center + egui::vec2(delta.x * cos - delta.y * sin, delta.x * sin + delta.y * cos)
    }

    fn video_rect_from_screen(&self, screen_rect: Rect, image_rect: Rect) -> Rect {
        let scale_x = self.video_size[0] as f32 / image_rect.width().max(1.0);
        let scale_y = self.video_size[1] as f32 / image_rect.height().max(1.0);
        Rect::from_min_max(
            pos2(
                ((screen_rect.left() - image_rect.left()) * scale_x)
                    .clamp(0.0, self.video_size[0] as f32),
                ((screen_rect.top() - image_rect.top()) * scale_y)
                    .clamp(0.0, self.video_size[1] as f32),
            ),
            pos2(
                ((screen_rect.right() - image_rect.left()) * scale_x)
                    .clamp(0.0, self.video_size[0] as f32),
                ((screen_rect.bottom() - image_rect.top()) * scale_y)
                    .clamp(0.0, self.video_size[1] as f32),
            ),
        )
    }

    fn screen_to_video(&self, point: Pos2, image_rect: Rect) -> Pos2 {
        let unrotated = self.rotate_screen_point(point, image_rect, -self.rotation_angle);
        self.video_rect_from_screen(
            Rect::from_min_size(unrotated, egui::vec2(0.0, 0.0)),
            image_rect,
        )
        .min
    }

    fn selection_from_screen(&self, start: Pos2, end: Pos2, image_rect: Rect) -> SelectionRect {
        let view_rect = Rect::from_points(&[start, end]);
        SelectionRect {
            corners: [
                self.screen_to_video(view_rect.left_top(), image_rect),
                self.screen_to_video(view_rect.right_top(), image_rect),
                self.screen_to_video(view_rect.right_bottom(), image_rect),
                self.screen_to_video(view_rect.left_bottom(), image_rect),
            ],
        }
    }

    fn selection_to_screen(&self, selection: SelectionRect, image_rect: Rect) -> [Pos2; 4] {
        let scale_x = image_rect.width() / self.video_size[0].max(1) as f32;
        let scale_y = image_rect.height() / self.video_size[1].max(1) as f32;
        selection.corners.map(|point| {
            self.rotate_screen_point(
                pos2(
                    image_rect.left() + point.x * scale_x,
                    image_rect.top() + point.y * scale_y,
                ),
                image_rect,
                self.rotation_angle,
            )
        })
    }

    fn selection_size(&self, selection: SelectionRect) -> (f32, f32) {
        (
            selection.corners[0].distance(selection.corners[1]),
            selection.corners[0].distance(selection.corners[3]),
        )
    }

    fn selection_gpu_rect(selection: SelectionRect) -> Option<[f32; 8]> {
        let origin = selection.corners[0];
        let edge_x = selection.corners[1] - origin;
        let edge_y = selection.corners[3] - origin;
        let width = edge_x.length();
        let height = edge_y.length();
        if width <= 0.0 || height <= 0.0 {
            return None;
        }
        let unit_x = edge_x / width;
        let unit_y = edge_y / height;
        Some([
            origin.x, origin.y, unit_x.x, unit_x.y, unit_y.x, unit_y.y, width, height,
        ])
    }

    fn update_gpu_rects(&mut self, image_rect: Rect) {
        let mut rects = self.rects.clone();
        if let (Some(begin), Some(motion)) = (self.rect_begin, self.rect_motion) {
            rects.push(self.selection_from_screen(begin, motion, image_rect));
        }

        self.gpu_rects = rects
            .into_iter()
            .filter_map(Self::selection_gpu_rect)
            .collect();

        let required = (self.gpu_rects.len().max(1) * 32) as u64;
        if required > self.inverse_rect_capacity {
            self.inverse_rect_buffer =
                self.render_state
                    .device
                    .create_buffer(&eframe::wgpu::BufferDescriptor {
                        label: Some("inverse_rects"),
                        size: required.next_power_of_two(),
                        usage: eframe::wgpu::BufferUsages::STORAGE
                            | eframe::wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
            self.inverse_rect_capacity = required.next_power_of_two();
            self.rgb_bind_group = None;
            self.yuyv_bind_group = None;
        }

        if !self.gpu_rects.is_empty() {
            let bytes = self
                .gpu_rects
                .iter()
                .flat_map(|rect| rect.iter().flat_map(|value| value.to_ne_bytes()))
                .collect::<Vec<_>>();
            self.render_state
                .queue
                .write_buffer(&self.inverse_rect_buffer, 0, &bytes);
        }
    }

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
        self.ratio.result.rotation_angle = self.rotation_angle;
        self.profiles.insert(active.clone(), self.ratio.result);
    }
    pub fn sync_from_profile(&mut self) {
        let active = self.active_profile.as_ref().unwrap();
        self.ratio.result = self.profiles[active];
        self.rotation_angle = self.ratio.result.rotation_angle;
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

impl UVCPlayer {
    pub fn capture_to_path(&self, target: &std::path::Path) -> Result<std::path::PathBuf> {
        let frame = self.shared_frame.read();
        if frame.seq == 0 {
            bail!("no frame available to capture");
        }

        let width = frame.width as u32;
        let height = frame.height as u32;
        if frame.rgb.len() < (width as usize) * (height as usize) * 3 {
            bail!("frame buffer too small");
        }

        let mut img: RgbaImage = RgbaImage::new(width, height);
        for y in 0..height {
            let start = (y as usize) * (width as usize) * 3;
            let src = &frame.rgb[start..start + (width as usize) * 3];
            for x in 0..width as usize {
                let si = x * 3;
                let px = image::Rgba([src[si], src[si + 1], src[si + 2], 255]);
                img.put_pixel(x as u32, y, px);
            }
        }

        // Target is always a directory; error if it exists and is a file
        if target.exists() && !target.is_dir() {
            bail!("target path exists but is not a directory: {:?}", target);
        }
        std::fs::create_dir_all(target)?;

        let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let out_path = target.join(format!("capture-{}.png", millis));

        img.save_with_format(&out_path, image::ImageFormat::Png)?;
        Ok(out_path)
    }
}

impl App for UVCPlayer {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        if ctx.input(|i| i.viewport().close_requested()) {
            let _ = self.shutdown_tx.send(true);
        }

        egui::containers::Panel::right("rpanel").show(ui, |ui| {
            ui.add_space(10.);

            // Camera selection UI
            ui.add(Label::new(
                RichText::new("Cameras").color(Color32::WHITE.gamma_multiply(0.9)),
            ));
            ui.label(&self.dmabuf_status);
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

            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.add(Label::new(
                    RichText::new("Capture").color(Color32::WHITE.gamma_multiply(0.9)),
                ));
                ui.add_space(6.);

                ui.add(
                    TextEdit::singleline(&mut self.save_path).desired_width(ui.available_width()),
                );
                if ui.button("Capture Frame").clicked() {
                    let path = std::path::Path::new(&self.save_path);
                    match self.capture_to_path(path) {
                        Ok(p) => self.last_capture_status = Some(format!("Saved {}", p.display())),
                        Err(e) => {
                            eprintln!("capture error: {:?}", e);
                            self.last_capture_status = Some(format!("Error: {}", e));
                        }
                    }
                }
                if let Some(msg) = &self.last_capture_status {
                    ui.label(msg);
                }
            });

            ui.add_space(15.);
            ui.separator();
            ui.add_space(10.);

            ui.add(Label::new(
                RichText::new("Profiles").color(Color32::WHITE.gamma_multiply(0.9)),
            ));
            ui.add_space(5.);

            for (pb, _) in &self.profiles {
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
            ui.label(RichText::new("Video rotation").color(Color32::WHITE.gamma_multiply(0.9)));
            if self.rotation_calibrating {
                ui.label("Drag a line over a feature that should be horizontal.");
                if ui.button("Cancel rotation").clicked() {
                    self.rotation_calibrating = false;
                    self.rotation_begin = None;
                    self.rotation_motion = None;
                }
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Set rotation").clicked() {
                        self.rotation_calibrating = true;
                        self.rotation_begin = None;
                        self.rotation_motion = None;
                    }
                    if self.rotation_angle != 0.0 && ui.button("Reset").clicked() {
                        self.rotation_angle = 0.0;
                        self.sync_to_profile();
                        let _ = self.dump_profile();
                    }
                });
            }
            if self.rotation_angle != 0.0 {
                ui.label(format!(
                    "Rotation: {:.1}°",
                    self.rotation_angle.to_degrees()
                ));
            }

            ui.add_space(20.);
            ui.add(Label::new(
                "press ESC to clear boxes. \npress Z to hide labels",
            ));
        });
        CentralPanel::default().show(ui, |ui| {
            let has_selection = self
                .selected_camera_uri
                .try_read()
                .ok()
                .and_then(|s| s.clone())
                .is_some();

            #[cfg(target_os = "linux")]
            {
                if let Some(index) = self.dma_in_flight {
                    let poll_result = self.render_state.device.poll(eframe::wgpu::PollType::Poll);
                    if matches!(
                        poll_result,
                        Ok(eframe::wgpu::PollStatus::QueueEmpty)
                            | Ok(eframe::wgpu::PollStatus::WaitSucceeded)
                    ) {
                        self.dma_in_flight = None;
                        self.shared_frame.write().dma_requeue = Some(index);
                    } else {
                        ctx.request_repaint_after(Duration::from_millis(1));
                    }
                }
                if self.dma_in_flight.is_none() {
                    if let Some((fd, metadata)) = self.shared_frame.write().dma.take() {
                        let mut raw_metadata = metadata;
                        raw_metadata.width = metadata.width.saturating_mul(2);
                        raw_metadata.height = metadata.height;
                        match dmabuf::import_texture(
                            &self.render_state.device,
                            fd,
                            raw_metadata,
                            eframe::wgpu::TextureFormat::R8Unorm,
                        ) {
                            Ok(texture) => {
                                if self.video_size != [metadata.width, metadata.height] {
                                    let new_texture = self.render_state.device.create_texture(
                                        &eframe::wgpu::TextureDescriptor {
                                            label: Some("yuyv_rgba_output"),
                                            size: eframe::wgpu::Extent3d {
                                                width: metadata.width,
                                                height: metadata.height,
                                                depth_or_array_layers: 1,
                                            },
                                            mip_level_count: 1,
                                            sample_count: 1,
                                            dimension: eframe::wgpu::TextureDimension::D2,
                                            format: eframe::wgpu::TextureFormat::Rgba8Unorm,
                                            usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING
                                                | eframe::wgpu::TextureUsages::STORAGE_BINDING,
                                            view_formats: &[],
                                        },
                                    );
                                    let new_view = new_texture.create_view(&Default::default());
                                    self.render_state
                                        .renderer
                                        .write()
                                        .update_egui_texture_from_wgpu_texture(
                                            &self.render_state.device,
                                            &new_view,
                                            eframe::wgpu::FilterMode::Linear,
                                            self.video_texture_id,
                                        );
                                    self.video_texture = new_texture;
                                    self.video_texture_view = new_view;
                                    self.video_size = [metadata.width, metadata.height];
                                    self.yuyv_bind_group = None;
                                }
                                let raw_view = texture.create_view(&Default::default());
                                self.yuyv_texture = Some(texture);
                                self.yuyv_texture_view = Some(raw_view);
                                self.yuyv_bind_group = Some(
                                    self.render_state.device.create_bind_group(
                                        &eframe::wgpu::BindGroupDescriptor {
                                            label: Some("yuyv_compute_bind_group"),
                                            layout: &self.yuyv_bind_group_layout,
                                            entries: &[
                                                eframe::wgpu::BindGroupEntry {
                                                    binding: 0,
                                                    resource:
                                                        eframe::wgpu::BindingResource::TextureView(
                                                            self.yuyv_texture_view
                                                                .as_ref()
                                                                .unwrap(),
                                                        ),
                                                },
                                                eframe::wgpu::BindGroupEntry {
                                                    binding: 1,
                                                    resource:
                                                        eframe::wgpu::BindingResource::TextureView(
                                                            &self.video_texture_view,
                                                        ),
                                                },
                                                eframe::wgpu::BindGroupEntry {
                                                    binding: 2,
                                                    resource: self
                                                        .rgb_dimensions
                                                        .as_entire_binding(),
                                                },
                                                eframe::wgpu::BindGroupEntry {
                                                    binding: 3,
                                                    resource: self
                                                        .inverse_rect_buffer
                                                        .as_entire_binding(),
                                                },
                                            ],
                                        },
                                    ),
                                );
                                self.render_state.queue.write_buffer(
                                    &self.rgb_dimensions,
                                    0,
                                    &self.dimensions_bytes(metadata.width, metadata.height),
                                );
                                let mut encoder = self.render_state.device.create_command_encoder(
                                    &eframe::wgpu::CommandEncoderDescriptor {
                                        label: Some("yuyv_to_rgba_encoder"),
                                    },
                                );
                                {
                                    let mut pass = encoder.begin_compute_pass(
                                        &eframe::wgpu::ComputePassDescriptor {
                                            label: Some("yuyv_to_rgba_pass"),
                                            timestamp_writes: None,
                                        },
                                    );
                                    pass.set_pipeline(&self.yuyv_pipeline);
                                    pass.set_bind_group(
                                        0,
                                        self.yuyv_bind_group.as_ref().unwrap(),
                                        &[],
                                    );
                                    pass.dispatch_workgroups(
                                        metadata.width.div_ceil(8),
                                        metadata.height.div_ceil(8),
                                        1,
                                    );
                                }
                                self.render_state.queue.submit(Some(encoder.finish()));
                                self.dma_in_flight = Some(metadata.buffer_index);
                                ctx.request_repaint();
                            }
                            Err(error) => {
                                self.shared_frame.write().dma_requeue = Some(metadata.buffer_index);
                                eprintln!(
                                    "render: DMA-BUF texture import failed for buffer {} \
                                 ({}x{}, fourcc=0x{:08x}, stride={}, offset={}): {error}",
                                    metadata.buffer_index,
                                    metadata.width,
                                    metadata.height,
                                    metadata.fourcc,
                                    metadata.stride,
                                    metadata.offset
                                );
                            }
                        }
                    }
                }
            }

            {
                let frame = self.shared_frame.try_read();
                if let Some(frame) = frame {
                    if frame.seq != 0 && frame.seq != self.last_frame_seq && {
                        #[cfg(target_os = "linux")]
                        {
                            self.dma_in_flight.is_none()
                        }
                        #[cfg(not(target_os = "linux"))]
                        {
                            true
                        }
                    } {
                        self.last_frame_seq = frame.seq;

                        if self.video_size != [frame.width, frame.height] {
                            let device = &self.render_state.device;
                            let new_texture =
                                device.create_texture(&eframe::wgpu::TextureDescriptor {
                                    label: Some("video_texture"),
                                    size: eframe::wgpu::Extent3d {
                                        width: frame.width,
                                        height: frame.height,
                                        depth_or_array_layers: 1,
                                    },
                                    mip_level_count: 1,
                                    sample_count: 1,
                                    dimension: eframe::wgpu::TextureDimension::D2,
                                    format: eframe::wgpu::TextureFormat::Rgba8Unorm,
                                    usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING
                                        | eframe::wgpu::TextureUsages::STORAGE_BINDING
                                        | eframe::wgpu::TextureUsages::COPY_DST,
                                    view_formats: &[],
                                });
                            let new_view = new_texture.create_view(&Default::default());
                            {
                                let mut renderer = self.render_state.renderer.write();
                                renderer.update_egui_texture_from_wgpu_texture(
                                    device,
                                    &new_view,
                                    eframe::wgpu::FilterMode::Linear,
                                    self.video_texture_id,
                                );
                            }
                            self.video_texture = new_texture;
                            self.video_texture_view = new_view;
                            self.video_size = [frame.width, frame.height];
                            self.rgb_bind_group = None;
                        }

                        let required =
                            (frame.width as u64 * frame.height as u64 * 3).next_multiple_of(4);
                        if required > self.rgb_buffer_capacity {
                            self.rgb_buffer = self.render_state.device.create_buffer(
                                &eframe::wgpu::BufferDescriptor {
                                    label: Some("packed_rgb24"),
                                    size: required,
                                    usage: eframe::wgpu::BufferUsages::STORAGE
                                        | eframe::wgpu::BufferUsages::COPY_DST,
                                    mapped_at_creation: false,
                                },
                            );
                            self.rgb_buffer_capacity = required;
                            self.rgb_bind_group = None;
                        }
                        self.render_state
                            .queue
                            .write_buffer(&self.rgb_buffer, 0, &frame.rgb);
                        self.render_state.queue.write_buffer(
                            &self.rgb_dimensions,
                            0,
                            &self.dimensions_bytes(frame.width, frame.height),
                        );
                        if self.rgb_bind_group.is_none() {
                            self.rgb_bind_group = Some(self.render_state.device.create_bind_group(
                                &eframe::wgpu::BindGroupDescriptor {
                                    label: Some("rgb24_compute_bind_group"),
                                    layout: &self.rgb_bind_group_layout,
                                    entries: &[
                                        eframe::wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: self.rgb_buffer.as_entire_binding(),
                                        },
                                        eframe::wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: eframe::wgpu::BindingResource::TextureView(
                                                &self.video_texture_view,
                                            ),
                                        },
                                        eframe::wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: self.rgb_dimensions.as_entire_binding(),
                                        },
                                        eframe::wgpu::BindGroupEntry {
                                            binding: 3,
                                            resource: self.inverse_rect_buffer.as_entire_binding(),
                                        },
                                    ],
                                },
                            ));
                        }
                        let bind_group = self.rgb_bind_group.as_ref().expect("bind group created");
                        let mut encoder = self.render_state.device.create_command_encoder(
                            &eframe::wgpu::CommandEncoderDescriptor {
                                label: Some("rgb24_to_rgba_encoder"),
                            },
                        );
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&eframe::wgpu::ComputePassDescriptor {
                                    label: Some("rgb24_to_rgba_pass"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(&self.rgb_pipeline);
                            pass.set_bind_group(0, bind_group, &[]);
                            pass.dispatch_workgroups(
                                frame.width.div_ceil(8),
                                frame.height.div_ceil(8),
                                1,
                            );
                        }
                        self.render_state.queue.submit(Some(encoder.finish()));
                        ctx.request_repaint();
                    }
                }
            }

            if !has_selection {
                ui.centered_and_justified(|ui| ui.label("select a camera"));
            } else {
                let response = ui.add(
                    Image::new(SizedTexture::new(
                        self.video_texture_id,
                        [self.video_size[0] as f32, self.video_size[1] as f32],
                    ))
                    .maintain_aspect_ratio(true)
                    .shrink_to_fit()
                    .rotate(self.rotation_angle, egui::vec2(0.5, 0.5)),
                );

                let pt = ui.painter().clone();
                let sense = response.interact(Sense::all());

                ui.input(|k| {
                    if k.key_pressed(egui::Key::Escape) {
                        self.rects.clear();
                        self.rotation_calibrating = false;
                        self.rotation_begin = None;
                        self.rotation_motion = None;
                    }
                    if k.key_pressed(egui::Key::Z) {
                        self.show_labels = !self.show_labels;
                    }
                });

                if sense.drag_started() {
                    if let Some(p) = sense.interact_pointer_pos() {
                        if self.rotation_calibrating {
                            self.rotation_begin = Some(p);
                        } else {
                            self.rect_begin = Some(p);
                        }
                    }
                }

                if sense.dragged() {
                    let _ = sense.drag_motion();
                    if let Some(p) = sense.interact_pointer_pos() {
                        if self.rotation_calibrating {
                            self.rotation_motion = Some(p);
                        } else {
                            self.rect_motion = Some(p);
                        }
                    }
                }
                if self.rotation_calibrating {
                    if let (Some(begin), Some(motion)) = (self.rotation_begin, self.rotation_motion)
                    {
                        if sense.drag_stopped() {
                            let delta = motion - begin;
                            if delta.length_sq() > 1.0 {
                                let mut angle = -delta.y.atan2(delta.x);
                                if angle > core::f32::consts::FRAC_PI_2 {
                                    angle -= core::f32::consts::PI;
                                } else if angle < -core::f32::consts::FRAC_PI_2 {
                                    angle += core::f32::consts::PI;
                                }
                                self.rotation_angle += angle;
                                self.rotation_angle = (self.rotation_angle + core::f32::consts::PI)
                                    .rem_euclid(2.0 * core::f32::consts::PI)
                                    - core::f32::consts::PI;
                                self.sync_to_profile();
                                let _ = self.dump_profile();
                            }
                            self.rotation_calibrating = false;
                            self.rotation_begin = None;
                            self.rotation_motion = None;
                        }
                    }
                }
                if let Some(bg) = self.rect_begin {
                    if let Some(mv) = self.rect_motion {
                        let moving_rect = self.selection_from_screen(bg, mv, response.rect);
                        let (width, height) = self.selection_size(moving_rect);

                        if self.ratio.calibrating {
                            let len = match self.ratio.calibrate_by {
                                CalibrationAxis::Width => width,
                                CalibrationAxis::Height => height,
                            };
                            if len > 0.0 {
                                self.ratio
                                    .result
                                    .calibrate(self.ratio.active.unwrap(), len as f64);
                            }
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

                self.update_gpu_rects(response.rect);
                let mut all_rects = self.rects.clone();
                if let (Some(begin), Some(motion)) = (self.rect_begin, self.rect_motion) {
                    all_rects.push(self.selection_from_screen(begin, motion, response.rect));
                }

                let dist_label = |px: f32| {
                    let show_mm_equivalence = self.ratio.calibrating
                        || self.ratio.active == Some(Magnification::Mm4)
                            && self.ratio.result.map[Magnification::Mm4] > 0.0;
                    if show_mm_equivalence && self.ratio.active.is_some() {
                        let n = self.ratio.from_px(px as f64) / 1_000.0;
                        format!("{:.2} mm = {:.0} px", n, px)
                    } else if self.ratio.active.is_some() {
                        let n = self.ratio.from_px(px as f64);
                        format!("{:.2}µm", n)
                    } else {
                        format!("{}px", px.round())
                    }
                };

                for rect in all_rects {
                    let screen_points = self.selection_to_screen(rect, response.rect);
                    if self.show_labels {
                        let screen_rect = Rect::from_points(&screen_points);
                        let rect_lb_x = Rect::EVERYTHING
                            .with_min_x(screen_rect.left() - 100.)
                            .with_max_x(screen_rect.right() + 100.)
                            .with_min_y(screen_rect.top() - 60.)
                            .with_max_y(screen_rect.top() - 8.);
                        let (width, height) = self.selection_size(rect);
                        let wd = dist_label(width);
                        let lb = RichText::new(wd)
                            .color(Color32::WHITE)
                            .size(40.)
                            .background_color(Color32::BLACK.gamma_multiply(0.2));
                        let lb = Label::new(lb).wrap_mode(TextWrapMode::Extend);
                        let rect_lb_y = Rect::EVERYTHING
                            .with_min_x(screen_rect.right() + 8.)
                            .with_min_y(screen_rect.top())
                            .with_max_y(screen_rect.bottom())
                            .with_max_x(screen_rect.right() + 280.);

                        ui.put(rect_lb_x, lb);
                        let ht = dist_label(height);
                        let lb = RichText::new(ht)
                            .color(Color32::WHITE)
                            .size(40.)
                            .background_color(Color32::BLACK.gamma_multiply(0.2));
                        let lb = Label::new(lb).wrap_mode(TextWrapMode::Extend);

                        ui.put(rect_lb_y, lb);
                    }
                }

                if let Some(pos) = ctx.pointer_latest_pos() {
                    ui.painter().add(epaint::PathShape::line(
                        vec![pos2(pos.x, 0.), pos2(pos.x, ctx.content_rect().height())],
                        PathStroke::new(3., Color32::WHITE),
                    ));
                    ui.painter().add(epaint::PathShape::line(
                        vec![pos2(0., pos.y), pos2(ctx.content_rect().width(), pos.y)],
                        PathStroke::new(3., Color32::WHITE),
                    ));
                }
                if self.rotation_calibrating {
                    if let (Some(begin), Some(motion)) = (self.rotation_begin, self.rotation_motion)
                    {
                        ui.painter()
                            .line_segment([begin, motion], Stroke::new(4.0, Color32::YELLOW));
                    }
                }
            }
        });
    }
}
