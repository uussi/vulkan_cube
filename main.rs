use glam::{Mat4, Vec3};
use std::{error::Error, sync::Arc, vec};
use vulkano::{
    buffer::{subbuffer, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{allocator::StandardDescriptorSetAllocator, WriteDescriptorSet},
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
struct Position {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

#[derive(BufferContents, Vertex, Default, Debug, Clone)]
#[repr(C)]
pub struct DummyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

impl DummyVertex {
    pub fn list() -> [DummyVertex; 6] {
        [
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [-1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, -1.0],
            },
        ]
    }
}

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}
#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
struct Mvp {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}

impl Mvp {
    fn new() -> Mvp {
        Mvp {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            view: glam::Mat4::IDENTITY.to_cols_array_2d(),
            projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    dummy_vertex: Subbuffer<[DummyVertex]>,
    index_buffer: Subbuffer<[u32]>,
    rcx: Option<RenderContext>,
    time_e: std::time::Instant,
    alloc: Arc<StandardMemoryAllocator>,
}

struct RenderContext {
    images: Vec<Arc<Image>>,
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    framebuffer: Option<Arc<ImageView>>,
    color_buffer: Option<Arc<ImageView>>,
    normal_buffer: Option<Arc<ImageView>>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline_01: Arc<GraphicsPipeline>,
    pipeline_02: Arc<GraphicsPipeline>,
    pipeline_03: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();

        let required_extensions = Surface::required_extensions(event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,

                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::default()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,

                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let obj_file = "./monkey.obj";


        let (models, _materials) = tobj::load_obj(
            &obj_file,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )
        .expect("Failed to OBJ load file");
        let m = models.first().unwrap();
        let mesh = &m.mesh;

        println!("{:?}", mesh.positions.len());
        // thread::sleep(Duration::from_millis(100));
        // Normals and texture coordinates are also loaded, but not printed in
        // this example.

        assert!(mesh.positions.len() % 3 == 0);


        // for vtx in 0..mesh.positions.len() / 3 {
        //         let normal = if !mesh.normals.is_empty() && 3 * vtx + 2 < mesh.normals.len() {
        //             [
        //                 mesh.normals[3 * vtx],
        //                 mesh.normals[3 * vtx + 1],
        //                 mesh.normals[3 * vtx + 2],
        //             ]
        //         } else {
        //             [1.0, 1.0, 1.0] // Default normal if none available
        //         };
        //     let temp = MyVertex {
        //         normal,
        //         position: [
        //             mesh.positions[3 * vtx],
        //             mesh.positions[3 * vtx + 1],
        //             mesh.positions[3 * vtx + 2],
        //         ],
        //          color: [1.0,1.0,1.0]
        //     };
        //     vertices.push(temp);
        // }
        let mut vertex_offset = 0;
        let mut index: Vec<u32> = Vec::new();
        let mut vertices = Vec::new();

        
        for m in &models {
            let mesh = &m.mesh;
            for vtx in 0..mesh.positions.len() / 3 {
                let normal = if !mesh.normals.is_empty() && 3 * vtx + 2 < mesh.normals.len() {
                    [
                        mesh.normals[3 * vtx],
                        mesh.normals[3 * vtx + 1],
                        mesh.normals[3 * vtx + 2],
                    ]
                } else {
                    [1.0, 1.0, 1.0]
                };
                vertices.push(MyVertex {
                    normal,
                    position: [
                        mesh.positions[3 * vtx],
                        mesh.positions[3 * vtx + 1],
                        mesh.positions[3 * vtx + 2],
                    ],
                    color: [1.0, 1.0, 1.0],
                });
            }
            for idx in &mesh.indices {
                index.push(idx + vertex_offset);
            }
            vertex_offset += (mesh.positions.len() / 3) as u32;
        }

        for vtx in 0..mesh.indices.len() {
            let temp = mesh.indices[vtx];
            index.push(temp);
        }

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            index,
        )
        .unwrap();

        let dummy_vertex = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DummyVertex::list().iter().cloned(),
        )
        .unwrap();

        let rcx = None;
        App {
            dummy_vertex,
            alloc: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            index_buffer,
            instance,
            device,
            queue,
            command_buffer_allocator,
            vertex_buffer,
            rcx,
            time_e: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_transparent(true)
                        .with_blur(true),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),

                    image_format,

                    image_extent: window_size.into(),

                    image_usage: ImageUsage::COLOR_ATTACHMENT,

                    composite_alpha: vulkano::swapchain::CompositeAlpha::PreMultiplied,

                    ..Default::default()
                },
            )
            .unwrap()
        };
        //----------------------------------------------------------------------
        //----------------------------------------------------------------------
        //----------------------------------------------------------------------
        //----------------------------------------------------------------------
        //----------------------------------------------------------------------
        mod deferred_vert {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "./shaders/deferred.vert",
            }
        }

        mod deferred_frag {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "./shaders/deferred.frag"
            }
        }

        mod directional_vert {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "./shaders/directional.vert",
            }
        }

        mod directional_frag {
            vulkano_shaders::shader! {
                ty: "fragment",
                path:  "./shaders/directional.frag",
            }
        }

        mod ambient_vert {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "./shaders/ambient.vert",
            }
        }

        mod ambient_frag {
            vulkano_shaders::shader! {
                ty: "fragment",
                path:  "./shaders/ambient.frag",
            }
        }

        let deferred_vert = deferred_vert::load(self.device.clone()).unwrap();
        let deferred_frag = deferred_frag::load(self.device.clone()).unwrap();

        let directional_vert = directional_vert::load(self.device.clone()).unwrap();
        let directional_frag = directional_frag::load(self.device.clone()).unwrap();

        let ambient_vert = ambient_vert::load(self.device.clone()).unwrap();
        let ambient_frag = ambient_frag::load(self.device.clone()).unwrap();

        let render_pass = vulkano::ordered_passes_renderpass!(
            self.device.clone(),
            attachments: {

                final_color:{
                    format:swapchain.image_format(),
                    samples: 1,
                    load_op:Clear,
                    store_op:Store,
                },
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op:DontCare,
                },
                normals: {
                    format: Format::R8G8B8A8_UNORM,
                    samples:1,
                    load_op:Clear,
                    store_op:DontCare,
                },
                depth:{
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                }
            },
            passes: [
            {

                color: [color,normals],
                depth_stencil: {depth},
                input:[],
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: [color, normals]
            }

        ]

        )
        .unwrap(); //}}}
                   // let vertices = [{{{
        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        let (framebuffers, color_buffer, normal_buffer) = window_size_dependent_setup(
            &images,
            &render_pass,
            self.device.clone(),
            self.alloc.clone(),
            swapchain.clone(),
        );
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------
        //--------------------------------------------------------------------

        let pipeline_01 = {
            //{{{
            let vs = deferred_vert.entry_point("main").unwrap();
            let fs = deferred_frag.entry_point("main").unwrap();

            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),

                    vertex_input_state: Some(vertex_input_state),

                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::TriangleList,
                        ..Default::default()
                    }),
                    viewport_state: Some(ViewportState {
                        viewports: [Viewport {
                            offset: [0.0, 0.0],
                            extent: window_size.into(),
                            depth_range: 0.0..=1.0,
                        }]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    }),

                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::Back,
                        front_face: FrontFace::Clockwise,
                        ..Default::default()
                    }),

                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        deferred_pass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState {
                            write_enable: true,
                            compare_op: CompareOp::Less,
                        }),
                        ..Default::default()
                    }),

                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(deferred_pass.into()),
                    ..GraphicsPipelineCreateInfo::new(layout)
                },
            )
            .unwrap()
        };

        let pipeline_02 = {
            //{{{
            let vs = directional_vert.entry_point("main").unwrap();
            let fs = directional_frag.entry_point("main").unwrap();

            let vertex_input_state = DummyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),

                    vertex_input_state: Some(vertex_input_state),

                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [Viewport {
                            offset: [0.0, 0.0],
                            extent: window_size.into(),
                            depth_range: 0.0..=1.0,
                        }]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    }),

                    rasterization_state: Some(RasterizationState {
                        ..Default::default()
                    }),

                    multisample_state: Some(MultisampleState::default()),

                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                            }),
                            ..Default::default()
                        },
                    )),

                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(lighting_pass.clone().into()),
                    ..GraphicsPipelineCreateInfo::new(layout)
                },
            )
            .unwrap()
        }; //}}}

        let pipeline_03 = {
            let vs = ambient_vert.entry_point("main").unwrap();
            let fs = ambient_frag.entry_point("main").unwrap();

            let vertex_input_state = DummyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),

                    vertex_input_state: Some(vertex_input_state),

                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [Viewport {
                            offset: [0.0, 0.0],
                            extent: window_size.into(),
                            depth_range: 0.0..=1.0,
                        }]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    }),

                    rasterization_state: Some(RasterizationState::default()),

                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                            }),
                            ..Default::default()
                        },
                    )),

                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(lighting_pass.into()),
                    ..GraphicsPipelineCreateInfo::new(layout)
                },
            )
            .unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let recreate_swapchain = false;

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(RenderContext {
            images,
            window,
            swapchain,
            render_pass,
            framebuffer: None,
            normal_buffer: None,
            color_buffer: None,
            framebuffers,
            pipeline_01,
            pipeline_02,
            pipeline_03,
            viewport,
            recreate_swapchain,
            previous_frame_end,
        });

        self.rcx.as_mut().unwrap().color_buffer = Some(color_buffer.clone());
        self.rcx.as_mut().unwrap().normal_buffer = Some(normal_buffer.clone());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rcx.swapchain = new_swapchain;

                    rcx.images = new_images.clone();
                    let (framebuffers, color_buffer, normal_buffer) = window_size_dependent_setup(
                        &new_images,
                        &rcx.render_pass,
                        self.device.clone(),
                        self.alloc.clone(),
                        rcx.swapchain.clone(),
                    );

                    rcx.viewport.extent = window_size.into();

                    rcx.recreate_swapchain = false;
                    rcx.color_buffer = Some(color_buffer.clone());
                    rcx.normal_buffer = Some(normal_buffer.clone());
                    rcx.framebuffers = framebuffers;
                }

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                //-------------------------------------------

                //-------------------------------------------
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.0, 0.0, 0.0, 0.0].into()),
                                Some([0.0, 0.0, 0.0, 1.0].into()),
                                Some([0.0, 0.0, 0.0, 1.0].into()),
                                Some(1.0.into()),
                            ],

                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap()
                    .bind_index_buffer(self.index_buffer.clone())
                    .unwrap();

                let mut mvp = Mvp::new();
                let image_extent: [u32; 2] = rcx.window.inner_size().into();
                let aspect_ratio = (image_extent[0] as f32 / image_extent[1] as f32) / 1.;

                let proj =
                    Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.01, 200.0);
                let view = Mat4::look_at_rh(
                    Vec3::new(0.0, 0.0, 1.5),
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, -1.0, 0.0),
                );

                let rotation = self.time_e.elapsed().as_secs() as f64
                    + self.time_e.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                let rotation_y = Mat4::from_rotation_y((rotation) as f32);
                // let rotation_x = Mat4::from_rotation_x((rotation) as f32);
                // let rotation_z = Mat4::from_rotation_z((rotation) as f32);

                let tranlate = Mat4::from_translation(Vec3::new(0.0, 0.0, -2.5));

                mvp.projection = proj.to_cols_array_2d();
                mvp.view = view.to_cols_array_2d();
                let scale = Mat4::from_scale(Vec3::new(1.5, 1.5, 0.5));
                mvp.model = (tranlate * scale * rotation_y).to_cols_array_2d();

                let memory_allocator = self.alloc.clone();
                let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
                    self.device.clone(),
                    Default::default(),
                ));

                let uniform = Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    mvp,
                )
                .expect("somthing go wrong for create this");

                let light = AmbientLight {
                    color: [1.0, 1.0, 1.0],
                    intensity: 0.1,
                };

                let directional_light_r = DirectionalLight {
                    position: [20.0, -5.0, -4.0],
                    color: [1.0, 1.0, 1.0],
                };

                let deferred_layout = rcx.pipeline_01.layout().set_layouts().first().unwrap();
                let lighting_layout = rcx.pipeline_02.layout().set_layouts().first().unwrap();
                let ambient_layout = rcx.pipeline_03.layout().set_layouts().first().unwrap();

                let ambient_subbuffer = Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    light,
                );

                let descriptor_set_deferred = vulkano::descriptor_set::DescriptorSet::new(
                    descriptor_set_allocator.clone(),
                    deferred_layout.clone(),
                    [WriteDescriptorSet::buffer(0, uniform.clone())],
                    [],
                )
                .unwrap();

                builder
                    .bind_pipeline_graphics(rcx.pipeline_01.clone())
                    .unwrap();
                builder
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap();
                builder
                    .bind_descriptor_sets(
                        vulkano::pipeline::PipelineBindPoint::Graphics,
                        rcx.pipeline_01.layout().clone(),
                        0,
                        descriptor_set_deferred.clone(),
                    )
                    .unwrap();

                unsafe { builder.draw_indexed(self.vertex_buffer.len() as u32, 1, 0, 0, 0) }
                    .unwrap();
                builder
                    .next_subpass(SubpassEndInfo::new(), SubpassBeginInfo::new())
                    .unwrap();

                //-----------------------------directional light

                // {
                //
                //                 let direct_light_buf =generate_directional_buffer(directional_light_g,self.alloc.clone()) ;
                //   let descriptor_set_lighting = vulkano::descriptor_set::DescriptorSet::new(
                //                     descriptor_set_allocator.clone(),
                //                     lighting_layout.clone(),
                //                     [
                //                         WriteDescriptorSet::image_view(0, rcx.color_buffer.clone().unwrap()),
                //                         WriteDescriptorSet::image_view(1, rcx.normal_buffer.clone().unwrap()),
                //                         WriteDescriptorSet::buffer(2, direct_light_buf.as_ref().clone()),
                //                     ],
                //                     [],
                //                 ).unwrap();
                //
                // builder.bind_pipeline_graphics(rcx.pipeline_02.clone()).unwrap();
                //                     builder.bind_vertex_buffers(0,self.dummy_vertex.clone());
                // builder
                //     .bind_descriptor_sets(
                //         vulkano::pipeline::PipelineBindPoint::Graphics,
                //         rcx.pipeline_02.layout().clone(),
                //         0,
                //         descriptor_set_lighting.clone(),
                //     )
                //     .unwrap();
                //
                //                 unsafe { builder.draw_indexed(self.dummy_vertex.len() as u32, 1, 0, 0,0) }.unwrap();
                // };
                //

                {
                    let direct_light_buf =
                        generate_directional_buffer(directional_light_r, self.alloc.clone());
                    let descriptor_set_lighting = vulkano::descriptor_set::DescriptorSet::new(
                        descriptor_set_allocator.clone(),
                        lighting_layout.clone(),
                        [
                            WriteDescriptorSet::image_view(0, rcx.color_buffer.clone().unwrap()),
                            WriteDescriptorSet::image_view(1, rcx.normal_buffer.clone().unwrap()),
                            WriteDescriptorSet::buffer(2, direct_light_buf.as_ref().clone()),
                        ],
                        [],
                    )
                    .unwrap();

                    builder
                        .bind_pipeline_graphics(rcx.pipeline_02.clone())
                        .unwrap();
                    builder
                        .bind_vertex_buffers(0, self.dummy_vertex.clone())
                        .unwrap();
                    builder
                        .bind_descriptor_sets(
                            vulkano::pipeline::PipelineBindPoint::Graphics,
                            rcx.pipeline_02.layout().clone(),
                            0,
                            descriptor_set_lighting.clone(),
                        )
                        .unwrap();

                    unsafe { builder.draw(6_u32, 1, 0, 0) }.unwrap();
                };

                //-----------------------------end directional light
                let descriptor_set_ambient = vulkano::descriptor_set::DescriptorSet::new(
                    descriptor_set_allocator.clone(),
                    ambient_layout.clone(),
                    [
                        WriteDescriptorSet::image_view(0, rcx.color_buffer.clone().unwrap()),
                        WriteDescriptorSet::buffer(1, ambient_subbuffer.unwrap().clone()),
                    ],
                    [],
                )
                .unwrap();

                builder
                    .bind_pipeline_graphics(rcx.pipeline_03.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        vulkano::pipeline::PipelineBindPoint::Graphics,
                        rcx.pipeline_03.layout().clone(),
                        0,
                        descriptor_set_ambient,
                    )
                    .unwrap();
                builder
                    .bind_vertex_buffers(0, self.dummy_vertex.clone())
                    .unwrap();

                unsafe { builder.draw(6_u32, 1, 0, 0) }.unwrap();

                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.build().unwrap();

                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::new(rcx.swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],

    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

#[derive(Default, Debug, Clone, BufferContents)]
#[repr(C)]
struct AmbientLight {
    color: [f32; 3],
    intensity: f32,
}
#[derive(Default, Debug, Clone, BufferContents)]
#[repr(C)]
struct DirectionalLight {
    position: [f32; 3],
    color: [f32; 3],
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    _device: Arc<Device>,
    allocator: Arc<StandardMemoryAllocator>,
    rcx: Arc<Swapchain>,
) -> (Vec<Arc<Framebuffer>>, Arc<ImageView>, Arc<ImageView>) {
    let allocator = allocator;

    let all_info = AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
        ..Default::default()
    };

    let dimensions = images[0].extent();
    let info = ImageCreateInfo {
        image_type: vulkano::image::ImageType::Dim2d,
        extent: dimensions,
        format: Format::D32_SFLOAT,
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
        ..Default::default()
    };

    let depth_buffer =
        ImageView::new_default(Image::new(allocator.clone(), info, all_info.clone()).unwrap())
            .unwrap();

    let color_buffer = ImageCreateInfo {
        image_type: vulkano::image::ImageType::Dim2d,
        extent: dimensions,
        format: rcx.image_format(),
        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
        ..Default::default()
    };

    let color_bi = ImageView::new_default(
        Image::new(allocator.clone(), color_buffer, all_info.clone()).unwrap(),
    )
    .unwrap();

    let normal_buffer = ImageCreateInfo {
        image_type: vulkano::image::ImageType::Dim2d,
        extent: dimensions,
        format: Format::R8G8B8A8_UNORM,
        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
        ..Default::default()
    };
    let normal_bi = ImageView::new_default(
        Image::new(allocator.clone(), normal_buffer, all_info.clone()).unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        color_bi.clone(),
                        normal_bi.clone(),
                        depth_buffer.clone(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    (framebuffers, color_bi.clone(), normal_bi.clone())
}

fn generate_directional_buffer(
    light: DirectionalLight,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Arc<subbuffer::Subbuffer<DirectionalLight>> {
    let buff = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        light,
    )
    .unwrap();
    Arc::new(buff)
}
