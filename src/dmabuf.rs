//! Vulkan DMA-BUF texture import for the eframe WGPU device.

use std::os::fd::OwnedFd;

use anyhow::Result;

use eframe::wgpu;

/// Layout metadata for a single-plane V4L2 DMA-BUF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmaBufFrame {
    pub buffer_index: u32,
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub stride: u32,
    pub offset: u32,
}

/// Import a V4L2-exported DMA-BUF into the renderer's existing Vulkan device.
///
/// The file descriptor is consumed by Vulkan on success (and closed on failure).
/// The caller must ensure that the descriptor describes a single-plane image and
/// that its layout metadata matches the V4L2 buffer.
#[cfg(all(target_os = "linux", not(target_arch = "wasm32")))]
pub fn import_texture(
    device: &wgpu::Device,
    fd: OwnedFd,
    frame: DmaBufFrame,
    format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    let descriptor = wgpu::TextureDescriptor {
        label: Some("v4l2-dmabuf"),
        size: wgpu::Extent3d {
            width: frame.width,
            height: frame.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };

    let hal_descriptor = wgpu::hal::TextureDescriptor {
        label: Some("v4l2-dmabuf"),
        size: descriptor.size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: descriptor.dimension,
        format,
        usage: wgpu::TextureUses::RESOURCE,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    };

    let hal_device = unsafe {
        device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| anyhow::anyhow!("eframe is not using the Vulkan backend"))?
    };
    let hal_texture = unsafe {
        hal_device
            .texture_from_dmabuf_fd(
                fd,
                &hal_descriptor,
                frame.modifier,
                u64::from(frame.stride),
                u64::from(frame.offset),
            )
            .map_err(|error| anyhow::anyhow!("Vulkan DMA-BUF import failed: {error:?}"))?
    };

    Ok(unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            hal_texture,
            &descriptor,
            wgpu::TextureUses::RESOURCE,
        )
    })
}

#[cfg(not(all(target_os = "linux", not(target_arch = "wasm32"))))]
pub fn import_texture(
    _device: &wgpu::Device,
    _fd: OwnedFd,
    _frame: DmaBufFrame,
    _format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    anyhow::bail!("DMA-BUF import is only available on Linux Vulkan")
}

pub fn support_status(adapter: &wgpu::Adapter) -> String {
    let info = adapter.get_info();
    let features = adapter.features();
    format!(
        "GPU: {} ({:?}); Vulkan DMA-BUF/YUYV-R8: {}",
        info.name,
        info.backend,
        features.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    )
}

pub fn is_supported(adapter: &wgpu::Adapter) -> bool {
    let info = adapter.get_info();
    info.backend == wgpu::Backend::Vulkan
        && adapter
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
}
