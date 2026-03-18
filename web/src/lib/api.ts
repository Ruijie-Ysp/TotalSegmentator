import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:28000';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5分钟超时（分割可能很慢）
});

// 类型定义
export interface SessionInfo {
  session_id: string;
  filename: string;
  status: 'uploaded' | 'processing' | 'completed' | 'error' | 'expired';
  shape?: number[];
  task?: string;
  organs?: string[];
  error?: string;
  file_type?: string;
  device_used?: string;
  debug_info?: { debug_dir?: string; summary?: string } | null;
  data_available?: boolean;
}

export interface SystemInfo {
  status: string;
  device: {
    cuda_available: boolean;
    mps_available: boolean;
    recommended_device: string;
    cuda_device_count?: number;
    cuda_device_name?: string;
  };
  supported_formats: string[];
  supported_2d_extensions: string[];
  timestamp: string;
}

export interface SliceInfo {
  axial: { max: number; default: number };
  sagittal: { max: number; default: number };
  coronal: { max: number; default: number };
}

export interface WindowPreset {
  center: number;
  width: number;
  name: string;
}

export interface TaskInfo {
  organ_count: number;
  organs: Record<number, string>;
}

// 任务状态信息（用于轮询进度）
export interface TaskStatus {
  session_id: string;
  status: 'uploaded' | 'processing' | 'completed' | 'error';
  progress: number;  // 0-100
  progress_message: string;
  filename?: string;
  task_type?: string;
  organs?: string[];
  shape?: number[];
  device_used?: string;
  error_message?: string;
}

// 活跃任务信息
export interface ActiveTask {
  session_id: string;
  filename: string;
  file_type?: string;
  status: string;
  task_type?: string;
  created_at?: string;
  progress: number;
  progress_message: string;
}

// API 函数
export async function uploadFile(file: File): Promise<{ session_id: string; filename: string }> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/api/upload', formData);
  return response.data;
}

export async function runSegmentation(
  sessionId: string,
  task: string = 'total',
  fast: boolean = true,
  device: string = 'auto',
  debugMode: boolean = false
): Promise<SessionInfo> {
  const response = await api.post(`/api/segment/${sessionId}`, {
    task,
    fast,
    device,
    debug_mode: debugMode,
  });
  return response.data;
}

export async function getSystemInfo(): Promise<SystemInfo> {
  const response = await api.get('/api/system');
  return response.data;
}

export async function getMetadata(sessionId: string): Promise<SessionInfo> {
  const response = await api.get(`/api/metadata/${sessionId}`);
  return response.data;
}

export async function getSliceInfo(sessionId: string): Promise<SliceInfo> {
  const response = await api.get(`/api/slice_info/${sessionId}`);
  return response.data;
}

export async function getTasks(): Promise<Record<string, TaskInfo>> {
  const response = await api.get('/api/task-types');
  return response.data;
}

export async function getWindowPresets(): Promise<Record<string, WindowPreset>> {
  const response = await api.get('/api/window_presets');
  return response.data;
}

// 获取当前活跃（进行中）的任务
export async function getActiveTask(): Promise<{ active_task: ActiveTask | null }> {
  const response = await api.get('/api/tasks/active/current');
  return response.data;
}

// 获取任务状态（用于轮询进度）
export async function getTaskStatus(sessionId: string): Promise<TaskStatus> {
  const response = await api.get(`/api/tasks/${sessionId}/status`);
  return response.data;
}

export type OverlayMode = 'fill' | 'contour' | 'fill_contour';

export function getSliceUrl(
  sessionId: string,
  axis: 'axial' | 'sagittal' | 'coronal',
  sliceIdx: number,
  windowCenter: number,
  windowWidth: number,
  overlayOrgans: string[],
  overlayOpacity: number,
  overlayMode: OverlayMode = 'fill',
  contourThickness: number = 2
): string {
  const params = new URLSearchParams({
    axis,
    slice_idx: sliceIdx.toString(),
    window_center: windowCenter.toString(),
    window_width: windowWidth.toString(),
    overlay_organs: overlayOrgans.join(','),
    overlay_opacity: overlayOpacity.toString(),
    overlay_mode: overlayMode,
    contour_thickness: contourThickness.toString(),
  });
  return `${API_BASE}/api/slice/${sessionId}?${params.toString()}`;
}

export async function exportLabelStudio(sessionId: string): Promise<object> {
  const response = await api.get(`/api/export/labelstudio/${sessionId}`);
  return response.data;
}

export function getExportNiftiUrl(sessionId: string): string {
  return `${API_BASE}/api/export/nifti/${sessionId}`;
}

// Label Studio 离线导出（ZIP 包 - 需要配置 Local Storage）
export function getExportLabelStudioOfflineUrl(
  sessionId: string,
  options?: {
    sliceStep?: number;
    axis?: 'axial' | 'sagittal' | 'coronal';
    includeOverlay?: boolean;
  }
): string {
  const params = new URLSearchParams();
  if (options?.sliceStep) params.set('slice_step', options.sliceStep.toString());
  if (options?.axis) params.set('axis', options.axis);
  if (options?.includeOverlay !== undefined) params.set('include_overlay', options.includeOverlay.toString());
  return `${API_BASE}/api/labelstudio/export-offline/${sessionId}?${params.toString()}`;
}

// Label Studio 简化导出（Base64 嵌入，直接可用，无需配置 Storage）
export async function exportLabelStudioSimple(
  sessionId: string,
  options?: {
    sliceStep?: number;
    axis?: 'axial' | 'sagittal' | 'coronal';
    maxSlices?: number;
  }
): Promise<object[]> {
  const params = new URLSearchParams();
  if (options?.sliceStep) params.set('slice_step', options.sliceStep.toString());
  if (options?.axis) params.set('axis', options.axis);
  if (options?.maxSlices) params.set('max_slices', options.maxSlices.toString());
  const response = await api.get(`/api/labelstudio/export-simple/${sessionId}?${params.toString()}`);
  return response.data;
}

// 器官标签位置信息
export interface OrganLabel {
  organ: string;
  x: number;  // 归一化坐标 0-1
  y: number;
  area_ratio: number;
  color: number[];
}

export interface OrganLabelsResponse {
  labels: OrganLabel[];
  slice_size: [number, number];
}

export async function getOrganLabels(
  sessionId: string,
  axis: 'axial' | 'sagittal' | 'coronal',
  sliceIdx: number,
  organs: string[]
): Promise<OrganLabelsResponse> {
  const params = new URLSearchParams({
    axis,
    slice_idx: sliceIdx.toString(),
    organs: organs.join(','),
  });
  const response = await api.get(`/api/organ_labels/${sessionId}?${params.toString()}`);
  return response.data;
}

// ============ LLM 配置 API ============

export interface LLMConfig {
  enabled: boolean;
  provider: string;
  base_url: string;
  model: string;
  temperature: number;
  max_tokens: number;
  timeout: number;  // LLM 请求超时时间（秒）
  system_prompt: string;
  qa_prompt_template: string;
  has_api_key: boolean;
}

export async function getLLMConfig(): Promise<LLMConfig> {
  const response = await api.get('/api/config/llm');
  return response.data;
}

export async function setLLMConfig(config: Partial<LLMConfig> & { api_key?: string }): Promise<{ status: string; enabled: boolean; model: string; has_api_key: boolean }> {
  const response = await api.post('/api/config/llm', config);
  return response.data;
}

export interface LLMTestConfig {
  base_url?: string;
  api_key?: string;
  model?: string;
}

export async function testLLMConnection(config?: LLMTestConfig): Promise<{ status: string; message: string }> {
  const response = await api.post('/api/config/llm/test', config || {});
  return response.data;
}

// ============ 训练数据导出 API ============

export interface ExportConfig {
  format: 'sharegpt' | 'alpaca';
  num_samples: number;
  slice_step: number;
  axis: 'axial' | 'sagittal' | 'coronal';
  include_augmentation: boolean;
  use_llm: boolean;
}

export interface ExportResult {
  format: string;
  total_samples: number;
  session_id: string;
  task: string;
  detected_organs: string[];
  config: {
    slice_step: number;
    num_samples_per_slice: number;
    axis: string;
    include_augmentation: boolean;
    use_llm: boolean;
    llm_enabled: boolean;
    llm_model: string | null;
  };
  data: Array<{
    conversations?: Array<{ from: string; value: string }>;
    instruction?: string;
    input?: string;
    output?: string;
    images: string[];
    source: string;
  }>;
}

export async function exportTrainingData(
  sessionId: string,
  config: Partial<ExportConfig>
): Promise<ExportResult> {
  const params = new URLSearchParams();
  if (config.format) params.set('format', config.format);
  if (config.num_samples) params.set('num_samples', config.num_samples.toString());
  if (config.slice_step) params.set('slice_step', config.slice_step.toString());
  if (config.axis) params.set('axis', config.axis);
  if (config.include_augmentation !== undefined) params.set('include_augmentation', config.include_augmentation.toString());
  if (config.use_llm !== undefined) params.set('use_llm', config.use_llm.toString());

  const response = await api.get(`/api/export/llama-factory/${sessionId}?${params.toString()}`);
  return response.data;
}

// 预览数据项接口
export interface PreviewItem {
  id: string;
  slice_idx: number;
  axis: string;
  image_path: string;
  image_base64: string;
  seg_overlay_base64?: string;  // 分割叠加图
  organs: string[];
  question?: string;
  answer?: string;
  instruction?: string;
  input?: string;
  output?: string;
  source: string;
  selected: boolean;
}

// 预览结果接口
export interface PreviewResult {
  session_id: string;
  format: string;
  total_available: number;
  max_preview: number;
  config: {
    slice_step: number;
    num_samples_per_slice: number;
    axis: string;
    include_augmentation: boolean;
    use_llm: boolean;
  };
  preview: PreviewItem[];
}

// 获取训练数据预览
export async function getTrainingPreview(
  sessionId: string,
  config: Partial<ExportConfig> & { max_preview?: number }
): Promise<PreviewResult> {
  const params = new URLSearchParams();
  if (config.format) params.set('format', config.format);
  if (config.num_samples) params.set('num_samples', config.num_samples.toString());
  if (config.slice_step) params.set('slice_step', config.slice_step.toString());
  if (config.axis) params.set('axis', config.axis);
  if (config.include_augmentation !== undefined) params.set('include_augmentation', config.include_augmentation.toString());
  if (config.use_llm !== undefined) params.set('use_llm', config.use_llm.toString());
  if (config.max_preview) params.set('max_preview', config.max_preview.toString());

  // LLM 模式需要更长的超时时间（每个切片可能需要10-30秒）
  const timeoutMs = config.use_llm ? 600000 : 300000; // LLM模式10分钟，普通模式5分钟

  const response = await api.get(`/api/export/llama-factory/${sessionId}/preview?${params.toString()}`, {
    timeout: timeoutMs
  });
  return response.data;
}

// ============ 预览数据持久化 API ============

// 获取缓存的预览数据
export interface CachedPreviewResponse {
  exists: boolean;
  preview: PreviewResult | null;
  images_available?: boolean;
  error?: string;
}

export async function getCachedPreview(sessionId: string): Promise<CachedPreviewResponse> {
  const response = await api.get(`/api/preview/${sessionId}`);
  return response.data;
}

// 更新缓存的预览数据
export async function updateCachedPreview(
  sessionId: string,
  preview: PreviewItem[]
): Promise<{ status: string; updated_count: number }> {
  const response = await api.put(`/api/preview/${sessionId}`, { preview });
  return response.data;
}

// 删除缓存的预览数据
export async function deleteCachedPreview(sessionId: string): Promise<{ status: string }> {
  const response = await api.delete(`/api/preview/${sessionId}`);
  return response.data;
}

// ============ 会话活动状态 API ============

export type ActivityStatus = 'idle' | 'generating_preview' | 'exporting';

// 更新会话活动状态
export async function updateSessionActivity(
  sessionId: string,
  activityStatus: ActivityStatus
): Promise<{ status: string; activity_status: string }> {
  const response = await api.put(`/api/session/${sessionId}/activity`, {
    activity_status: activityStatus,
  });
  return response.data;
}

// 获取所有会话的活动状态
export interface SessionActivitiesResponse {
  activities: Record<string, { activity_status: ActivityStatus; updated_at: string }>;
}

export async function getSessionActivities(): Promise<SessionActivitiesResponse> {
  const response = await api.get('/api/sessions/activities');
  return response.data;
}

// ============ QA 模板配置 API ============

export interface QATemplateCategory {
  templates: string[];
  is_custom: boolean;
  enabled: boolean;   // 是否启用
  priority: boolean;  // 是否为重点（优先使用）
}

export interface QATemplatesResponse {
  categories: Record<string, QATemplateCategory>;
  available_categories: string[];
  description: Record<string, string>;
}

export interface QATemplatesUpdateRequest {
  organ_identification?: string[];
  organ_location?: string[];
  organ_description?: string[];
  diagnostic_analysis?: string[];
  comprehensive_report?: string[];
}

export interface TemplateSelectionRequest {
  category: string;
  enabled?: boolean;
  priority?: boolean;
}

export interface TemplateBatchSelectionRequest {
  selections: Record<string, { enabled: boolean; priority: boolean }>;
}

// 获取 QA 模板配置
export async function getQATemplates(): Promise<QATemplatesResponse> {
  const response = await api.get('/api/config/qa-templates');
  return response.data;
}

// 更新 QA 模板配置
export async function setQATemplates(templates: QATemplatesUpdateRequest): Promise<{ status: string; updated_categories: string[]; message: string }> {
  const response = await api.post('/api/config/qa-templates', templates);
  return response.data;
}

// 设置单个模板类别的选择状态
export async function setTemplateSelection(request: TemplateSelectionRequest): Promise<{ status: string; category: string; state: { enabled: boolean; priority: boolean } }> {
  const response = await api.post('/api/config/qa-templates/selection', request);
  return response.data;
}

// 批量设置模板选择状态
export async function setTemplateSelectionBatch(request: TemplateBatchSelectionRequest): Promise<{ status: string; updated_categories: string[]; message: string }> {
  const response = await api.post('/api/config/qa-templates/selection/batch', request);
  return response.data;
}

// 重置 QA 模板为默认值
export async function resetQATemplates(category?: string): Promise<{ status: string; category?: string; message?: string }> {
  const params = category ? `?category=${category}` : '';
  const response = await api.post(`/api/config/qa-templates/reset${params}`);
  return response.data;
}


// ==================== Label Studio 集成 API ====================

// Label Studio 集成状态
export interface LabelStudioStatus {
  labelstudio: {
    configured: boolean;
    url: string;
    connected: boolean;
    sdk_available: boolean;
  };
  minio: {
    endpoint: string;
    bucket: string;
    connected: boolean;
  };
}

// 同步结果
export interface LabelStudioSyncResult {
  success: boolean;
  message: string;
  project_id: number | null;
  tasks_imported: number;
  slices_uploaded: number;
}

// Label Studio 项目
export interface LabelStudioProject {
  id: number;
  title: string;
  task_count: number;
}

// 获取 Label Studio 集成状态
export async function getLabelStudioStatus(): Promise<LabelStudioStatus> {
  const response = await api.get('/api/labelstudio/status');
  return response.data;
}

// 同步会话到 Label Studio
export async function syncToLabelStudio(
  sessionId: string,
  options?: {
    projectName?: string;
    sliceStep?: number;
    axis?: 'axial' | 'sagittal' | 'coronal';
    organs?: string[];
  }
): Promise<LabelStudioSyncResult> {
  const params = new URLSearchParams();
  if (options?.projectName) params.set('project_name', options.projectName);
  if (options?.sliceStep) params.set('slice_step', options.sliceStep.toString());
  if (options?.axis) params.set('axis', options.axis);
  // organs 通过请求体传递，因为可能很长

  const response = await api.post(
    `/api/labelstudio/sync/${sessionId}?${params.toString()}`,
    { organs: options?.organs }
  );
  return response.data;
}

// 获取 Label Studio 项目列表
export async function getLabelStudioProjects(): Promise<LabelStudioProject[]> {
  const response = await api.get('/api/labelstudio/projects');
  return response.data;
}

// 上传切片到 MinIO
export async function uploadSlicesToMinio(
  sessionId: string,
  options?: {
    sliceStep?: number;
    axis?: 'axial' | 'sagittal' | 'coronal';
  }
): Promise<{ success: boolean; uploaded_count: number; slices: Array<{ slice_idx: number; public_url: string }> }> {
  const params = new URLSearchParams();
  if (options?.sliceStep) params.set('slice_step', options.sliceStep.toString());
  if (options?.axis) params.set('axis', options.axis);

  const response = await api.post(`/api/labelstudio/upload-slices/${sessionId}?${params.toString()}`);
  return response.data;
}

