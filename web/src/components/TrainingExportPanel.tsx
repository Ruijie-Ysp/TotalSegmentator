'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Settings, Download, Loader2, CheckCircle, XCircle, AlertCircle, Sparkles, Eye, ChevronLeft, ChevronRight, Check, X, FileText, RotateCcw, Plus, Trash2, Star, RefreshCw } from 'lucide-react';
import ZoomableImage from './ZoomableImage';
import { getOrganDisplayName } from '@/lib/organs';
import {
  getLLMConfig,
  setLLMConfig,
  testLLMConnection,
  getTrainingPreview,
  getQATemplates,
  setQATemplates,
  resetQATemplates,
  setTemplateSelection,
  getCachedPreview,
  updateCachedPreview,
  updateSessionActivity,
  LLMConfig,
  ExportConfig,
  PreviewItem,
  PreviewResult,
  QATemplatesResponse,
  ActivityStatus as ApiActivityStatus,
} from '@/lib/api';

// 本地定义的导出结果类型（直接从预览数据生成）
interface LocalExportResult {
  session_id: string;
  format: string;
  total_samples: number;
  config: {
    use_llm?: boolean;
    exported_from_preview?: boolean;
    [key: string]: unknown;
  };
  data: unknown[];
}

// 步骤类型
type Step = 'config' | 'preview' | 'export';

// 活动状态类型
export type ActivityStatus = 'idle' | 'generating_preview' | 'exporting';

interface TrainingExportPanelProps {
  sessionId: string | null;
  isOpen: boolean;
  onClose: () => void;
  onActivityChange?: (status: ActivityStatus, sessionId: string | null) => void;
}

export default function TrainingExportPanel({
  sessionId,
  isOpen,
  onClose,
  onActivityChange,
}: TrainingExportPanelProps) {
  // 当前步骤
  const [currentStep, setCurrentStep] = useState<Step>('config');

  // LLM 配置状态
  const [llmConfig, setLlmConfigState] = useState<LLMConfig | null>(null);
  const [apiKey, setApiKey] = useState('');
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [testMessage, setTestMessage] = useState('');

  // 导出配置状态
  const [exportConfig, setExportConfig] = useState<ExportConfig>({
    format: 'sharegpt',
    num_samples: 3,
    slice_step: 10,
    axis: 'axial',
    include_augmentation: true,
    use_llm: false,
  });

  // 预览状态
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewData, setPreviewData] = useState<PreviewResult | null>(null);
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [currentPreviewIndex, setCurrentPreviewIndex] = useState(0);
  const [previewError, setPreviewError] = useState<{message: string; details?: string; code?: string} | null>(null);

  // 导出状态
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<LocalExportResult | null>(null);
  const [exportError, setExportError] = useState<{message: string; details?: string} | null>(null);

  // QA 模板状态
  const [qaTemplates, setQaTemplates] = useState<QATemplatesResponse | null>(null);
  const [editingCategory, setEditingCategory] = useState<string | null>(null);
  const [editingTemplates, setEditingTemplates] = useState<string[]>([]);
  const [showTemplatePanel, setShowTemplatePanel] = useState(false);

  // 问答编辑状态
  const [editingQAId, setEditingQAId] = useState<string | null>(null);
  const [editingQuestion, setEditingQuestion] = useState('');
  const [editingAnswer, setEditingAnswer] = useState('');

  // 缓存加载状态
  const [loadingCache, setLoadingCache] = useState(false);
  const [hasCachedPreview, setHasCachedPreview] = useState(false);

  // 防抖保存 ref
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // 通知活动状态变化（同时更新后端）
  useEffect(() => {
    if (onActivityChange) {
      if (previewLoading) {
        onActivityChange('generating_preview', sessionId);
      } else if (exporting) {
        onActivityChange('exporting', sessionId);
      } else {
        onActivityChange('idle', sessionId);
      }
    }

    // 同步更新后端活动状态
    if (sessionId) {
      const status: ApiActivityStatus = previewLoading
        ? 'generating_preview'
        : exporting
          ? 'exporting'
          : 'idle';
      updateSessionActivity(sessionId, status).catch(console.error);
    }
  }, [previewLoading, exporting, sessionId, onActivityChange]);

  // 加载 LLM 配置、QA 模板和缓存的预览数据
  useEffect(() => {
    if (isOpen) {
      getLLMConfig().then(setLlmConfigState).catch(console.error);
      getQATemplates().then(setQaTemplates).catch(console.error);

      // 检查是否有缓存的预览数据
      if (sessionId) {
        setLoadingCache(true);
        getCachedPreview(sessionId)
          .then((result) => {
            if (result.exists && result.preview) {
              setHasCachedPreview(true);
              // 如果有图像数据，直接恢复预览
              if (result.images_available && result.preview.preview) {
                setPreviewData({
                  session_id: result.preview.session_id || sessionId,
                  format: result.preview.format || 'sharegpt',
                  total_available: result.preview.preview.length,
                  max_preview: result.preview.preview.length,
                  config: result.preview.config || {
                    slice_step: 10,
                    num_samples_per_slice: 3,
                    axis: 'axial',
                    include_augmentation: true,
                    use_llm: false,
                  },
                  preview: result.preview.preview,
                });
                // 默认全选
                const allIds = new Set(result.preview.preview.map((item: PreviewItem) => item.id));
                setSelectedItems(allIds);
                setCurrentStep('preview');
              }
            } else {
              setHasCachedPreview(false);
            }
          })
          .catch(console.error)
          .finally(() => setLoadingCache(false));
      }
    }
  }, [isOpen, sessionId]);
  
  // 保存 LLM 配置
  const handleSaveLLMConfig = useCallback(async () => {
    if (!llmConfig) return;

    try {
      const result = await setLLMConfig({
        enabled: llmConfig.enabled,
        base_url: llmConfig.base_url,
        model: llmConfig.model,
        temperature: llmConfig.temperature,
        max_tokens: llmConfig.max_tokens,
        timeout: llmConfig.timeout,
        api_key: apiKey || undefined,
      });

      setLlmConfigState(prev => prev ? { ...prev, has_api_key: result.has_api_key } : null);
      setApiKey('');
      alert('配置已保存');
    } catch (error) {
      console.error('保存配置失败:', error);
      alert('保存配置失败');
    }
  }, [llmConfig, apiKey]);
  
  // 测试 LLM 连接（使用当前编辑的配置）
  const handleTestConnection = useCallback(async () => {
    if (!llmConfig) return;

    setTestStatus('testing');
    try {
      // 传递当前编辑的配置，而不是后端保存的配置
      const result = await testLLMConnection({
        base_url: llmConfig.base_url,
        model: llmConfig.model,
        api_key: apiKey || undefined,  // 如果有新输入的 API Key，使用它
      });
      setTestStatus(result.status === 'success' ? 'success' : 'error');
      setTestMessage(result.message);
    } catch (error) {
      setTestStatus('error');
      setTestMessage('连接测试失败');
    }
  }, [llmConfig, apiKey]);
  
  // 生成预览
  const handleGeneratePreview = useCallback(async () => {
    if (!sessionId) return;

    setPreviewLoading(true);
    setPreviewError(null);

    try {
      const result = await getTrainingPreview(sessionId, {
        ...exportConfig,
        max_preview: 500, // 预览所有数据（实际数量由切片步长决定）
      });
      setPreviewData(result);
      // 默认全选
      const allIds = new Set(result.preview.map(item => item.id));
      setSelectedItems(allIds);
      setCurrentPreviewIndex(0);
      setCurrentStep('preview');
    } catch (error: unknown) {
      console.error('生成预览失败:', error);

      // 解析错误信息
      let errorMessage = '生成预览失败';
      let errorDetails = '';
      let errorCode = '';

      if (error instanceof Error) {
        errorMessage = error.message;

        // 检查是否是 Axios 错误
        const axiosError = error as { code?: string; response?: { data?: { detail?: string }; status?: number } };

        if (axiosError.code === 'ECONNABORTED' || error.message.includes('timeout')) {
          errorMessage = 'LLM 请求超时';
          errorDetails = '大模型响应时间过长。可能原因：\n• 网络连接不稳定\n• LLM 服务繁忙\n• 图像数据过大\n\n建议：减少每次预览的切片数量，或检查 LLM 服务状态。';
          errorCode = 'TIMEOUT';
        } else if (axiosError.response) {
          const status = axiosError.response.status;
          const detail = axiosError.response.data?.detail;

          errorCode = `HTTP ${status}`;
          errorDetails = detail || JSON.stringify(axiosError.response.data);

          if (status === 401) {
            errorMessage = 'LLM API 认证失败';
            errorDetails = 'API Key 无效或已过期，请检查 LLM 配置。';
          } else if (status === 429) {
            errorMessage = 'LLM API 请求限流';
            errorDetails = '请求过于频繁，请稍后再试。';
          } else if (status === 500) {
            errorMessage = 'LLM 服务内部错误';
            errorDetails = detail || '服务器内部错误，请检查后端日志。';
          } else if (status === 404) {
            errorMessage = '会话未找到';
            errorDetails = '可能需要重新上传文件并进行分割。';
          }
        } else if (error.message.includes('Network Error')) {
          errorMessage = '网络连接失败';
          errorDetails = '无法连接到后端服务，请检查：\n• 后端服务是否正在运行\n• 网络连接是否正常';
          errorCode = 'NETWORK_ERROR';
        }
      }

      setPreviewError({
        message: errorMessage,
        details: errorDetails,
        code: errorCode
      });
    } finally {
      setPreviewLoading(false);
    }
  }, [sessionId, exportConfig]);

  // 切换选中状态
  const toggleItemSelection = useCallback((id: string) => {
    setSelectedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  }, []);

  // 开始编辑问答对
  const startEditQA = useCallback((qa: PreviewItem) => {
    setEditingQAId(qa.id);
    setEditingQuestion(qa.question || '');
    setEditingAnswer(qa.answer || '');
  }, []);

  // 保存编辑的问答对
  const saveEditQA = useCallback(() => {
    if (!editingQAId || !previewData) return;

    // 更新预览数据中的问答对
    const updatedPreview = previewData.preview.map(item => {
      if (item.id === editingQAId) {
        return {
          ...item,
          question: editingQuestion,
          answer: editingAnswer,
          source: 'manual_edit' // 标记为手动编辑
        };
      }
      return item;
    });

    const newPreviewData = {
      ...previewData,
      preview: updatedPreview
    };

    setPreviewData(newPreviewData);

    // 持久化到后端（防抖处理）
    if (sessionId) {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
      saveTimeoutRef.current = setTimeout(() => {
        updateCachedPreview(sessionId, updatedPreview)
          .then(() => console.log('[预览] ✓ 已保存修改'))
          .catch((e) => console.error('[预览] ⚠ 保存失败:', e));
      }, 500);
    }

    // 清除编辑状态
    setEditingQAId(null);
    setEditingQuestion('');
    setEditingAnswer('');
  }, [editingQAId, editingQuestion, editingAnswer, previewData, sessionId]);

  // 取消编辑
  const cancelEditQA = useCallback(() => {
    setEditingQAId(null);
    setEditingQuestion('');
    setEditingAnswer('');
  }, []);

  // 清理定时器
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, []);

  // 导出训练数据 - 直接使用预览阶段已选择的数据
  const handleExport = useCallback(async () => {
    if (!sessionId || !previewData) return;

    setExporting(true);
    setExportResult(null);

    try {
      // 从预览数据中筛选已选择的项
      const selectedPreviewItems = previewData.preview.filter(item => selectedItems.has(item.id));

      // 转换为训练数据格式
      let trainingData: unknown[];
      if (exportConfig.format === 'sharegpt') {
        trainingData = selectedPreviewItems.map(item => ({
          conversations: [
            { from: "human", value: `<image>\n${item.question}` },
            { from: "gpt", value: item.answer }
          ],
          images: [item.image_path],
          source: item.source || "template"
        }));
      } else {
        // Alpaca 格式
        trainingData = selectedPreviewItems.map(item => ({
          instruction: item.question || (item as { instruction?: string }).instruction || "",
          input: "",
          output: item.answer || (item as { output?: string }).output || "",
          images: [item.image_path],
          source: item.source || "template"
        }));
      }

      // 构建导出结果
      const result = {
        session_id: sessionId,
        format: exportConfig.format,
        total_samples: trainingData.length,
        config: {
          ...exportConfig,
          exported_from_preview: true
        },
        data: trainingData
      };

      setExportResult(result);
      setCurrentStep('export');

      // 下载 JSON 文件
      const blob = new Blob([JSON.stringify(trainingData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `training_${sessionId}_${exportConfig.format}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('导出失败:', error);
      alert('导出失败');
    } finally {
      setExporting(false);
    }
  }, [sessionId, exportConfig, previewData, selectedItems]);

  // 重置面板
  const handleReset = useCallback(() => {
    setCurrentStep('config');
    setPreviewData(null);
    setSelectedItems(new Set());
    setExportResult(null);
  }, []);

  // 按切片分组预览数据 - 必须在所有条件返回之前定义
  interface SliceGroup {
    slice_idx: number;
    axis: string;
    image_base64: string;
    seg_overlay_base64?: string;
    image_path: string;
    organs: string[];
    qa_pairs: PreviewItem[];
  }

  const groupedPreviewData: SliceGroup[] = React.useMemo(() => {
    if (!previewData?.preview) return [];

    const groups: Map<string, SliceGroup> = new Map();

    for (const item of previewData.preview) {
      const key = `${item.axis}_${item.slice_idx}`;
      if (!groups.has(key)) {
        groups.set(key, {
          slice_idx: item.slice_idx,
          axis: item.axis,
          image_base64: item.image_base64,
          seg_overlay_base64: item.seg_overlay_base64,
          image_path: item.image_path,
          organs: item.organs,
          qa_pairs: []
        });
      }
      groups.get(key)!.qa_pairs.push(item);
    }

    return Array.from(groups.values());
  }, [previewData?.preview]);

  // 当前切片组
  const currentSliceGroup = groupedPreviewData[currentPreviewIndex];

  // 当前预览项（保留兼容性）
  const currentPreviewItem = previewData?.preview[currentPreviewIndex];

  // 条件返回必须在所有 hooks 之后
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-2">
      <div className="bg-gray-900 rounded-lg w-[95vw] h-[95vh] overflow-hidden flex flex-col">
        {/* 标题栏 + 步骤指示器 合并 */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            训练数据导出
          </h2>

          {/* 步骤指示器 - 居中 */}
          <div className="flex items-center gap-3">
            {[
              { key: 'config', label: '1. 配置', icon: Settings },
              { key: 'preview', label: '2. 预览审核', icon: Eye },
              { key: 'export', label: '3. 导出', icon: Download },
            ].map((step, index) => (
              <React.Fragment key={step.key}>
                {index > 0 && <div className="w-6 h-0.5 bg-gray-600" />}
                <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs ${
                  currentStep === step.key
                    ? 'bg-purple-600 text-white'
                    : step.key === 'export' && exportResult
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-700 text-gray-400'
                }`}>
                  <step.icon className="w-3.5 h-3.5" />
                  {step.label}
                </div>
              </React.Fragment>
            ))}
          </div>

          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl">✕</button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {/* 步骤1: 配置 */}
          {currentStep === 'config' && (
            <div className="space-y-6">
              {/* LLM 配置区域 */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                  <Settings className="w-4 h-4" /> LLM 配置（可选）
                </h3>
                <div className="bg-gray-800 rounded-lg p-3 space-y-3">
                  <div className="flex items-center gap-3">
                    <label className="flex items-center gap-2 text-sm text-gray-300">
                      <input
                        type="checkbox"
                        checked={llmConfig?.enabled || false}
                        onChange={(e) => setLlmConfigState(prev => prev ? { ...prev, enabled: e.target.checked } : null)}
                        className="rounded"
                      />
                      启用 LLM 生成
                    </label>
                    {llmConfig?.has_api_key && (
                      <span className="text-xs text-green-400 flex items-center gap-1">
                        <CheckCircle className="w-3 h-3" /> API Key 已配置
                      </span>
                    )}
                  </div>
                  {llmConfig?.enabled && (
                    <>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-xs text-gray-400">Base URL</label>
                          <input
                            type="text"
                            value={llmConfig?.base_url || ''}
                            onChange={(e) => setLlmConfigState(prev => prev ? { ...prev, base_url: e.target.value } : null)}
                            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                            placeholder="https://api.openai.com/v1"
                          />
                        </div>
                        <div>
                          <label className="text-xs text-gray-400">模型</label>
                          <input
                            type="text"
                            value={llmConfig?.model || ''}
                            onChange={(e) => setLlmConfigState(prev => prev ? { ...prev, model: e.target.value } : null)}
                            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                            placeholder="gpt-4o"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-xs text-gray-400">API Key（输入后保存）</label>
                          <input
                            type="password"
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                            placeholder="sk-..."
                          />
                        </div>
                        <div>
                          <label className="text-xs text-gray-400">请求超时（秒）</label>
                          <input
                            type="number"
                            min={30}
                            max={600}
                            value={llmConfig?.timeout || 120}
                            onChange={(e) => setLlmConfigState(prev => prev ? { ...prev, timeout: parseInt(e.target.value) || 120 } : null)}
                            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                            placeholder="120"
                          />
                          <span className="text-xs text-gray-500">每次请求成功后自动重置</span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button onClick={handleSaveLLMConfig} className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm text-white">保存配置</button>
                        <button onClick={handleTestConnection} disabled={testStatus === 'testing'} className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm text-white flex items-center gap-1">
                          {testStatus === 'testing' && <Loader2 className="w-3 h-3 animate-spin" />}
                          测试连接
                        </button>
                        {testStatus === 'success' && <span className="text-green-400 text-sm flex items-center gap-1"><CheckCircle className="w-4 h-4" />{testMessage}</span>}
                        {testStatus === 'error' && <span className="text-red-400 text-sm flex items-center gap-1"><XCircle className="w-4 h-4" />{testMessage}</span>}
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* 导出配置区域 */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                  <Download className="w-4 h-4" /> 导出配置
                </h3>
                <div className="bg-gray-800 rounded-lg p-3 space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-400">数据格式</label>
                      <select value={exportConfig.format} onChange={(e) => setExportConfig(prev => ({ ...prev, format: e.target.value as 'sharegpt' | 'alpaca' }))} className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white">
                        <option value="sharegpt">ShareGPT (对话格式)</option>
                        <option value="alpaca">Alpaca (指令格式)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-gray-400">切片轴向</label>
                      <select value={exportConfig.axis} onChange={(e) => setExportConfig(prev => ({ ...prev, axis: e.target.value as 'axial' | 'sagittal' | 'coronal' }))} className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white">
                        <option value="axial">轴状位 (Axial)</option>
                        <option value="sagittal">矢状位 (Sagittal)</option>
                        <option value="coronal">冠状位 (Coronal)</option>
                      </select>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-400">每切片问答数</label>
                      <input type="number" min={1} max={10} value={exportConfig.num_samples} onChange={(e) => setExportConfig(prev => ({ ...prev, num_samples: parseInt(e.target.value) || 3 }))} className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white" />
                    </div>
                    <div>
                      <label className="text-xs text-gray-400">切片采样步长</label>
                      <input type="number" min={1} max={50} value={exportConfig.slice_step} onChange={(e) => setExportConfig(prev => ({ ...prev, slice_step: parseInt(e.target.value) || 10 }))} className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white" />
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <label className="flex items-center gap-2 text-sm text-gray-300">
                      <input type="checkbox" checked={exportConfig.include_augmentation} onChange={(e) => setExportConfig(prev => ({ ...prev, include_augmentation: e.target.checked }))} className="rounded" />
                      数据增强
                    </label>
                    <label className="flex items-center gap-2 text-sm text-gray-300">
                      <input type="checkbox" checked={exportConfig.use_llm} onChange={(e) => setExportConfig(prev => ({ ...prev, use_llm: e.target.checked }))} disabled={!llmConfig?.enabled || !llmConfig?.has_api_key} className="rounded" />
                      使用 LLM 生成
                      {exportConfig.use_llm && <Sparkles className="w-3 h-3 text-purple-400" />}
                    </label>
                  </div>
                  {exportConfig.use_llm && (!llmConfig?.enabled || !llmConfig?.has_api_key) && (
                    <div className="text-xs text-yellow-400 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" />请先配置并启用 LLM
                    </div>
                  )}
                </div>
              </div>

              {/* QA 模板配置区域 */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                    <FileText className="w-4 h-4" /> 问答模板配置
                  </h3>
                  <button
                    onClick={() => setShowTemplatePanel(!showTemplatePanel)}
                    className="text-xs text-blue-400 hover:text-blue-300"
                  >
                    {showTemplatePanel ? '收起' : '展开配置'}
                  </button>
                </div>

                {showTemplatePanel && qaTemplates && (
                  <div className="bg-gray-800 rounded-lg p-3 space-y-3">
                    <div className="text-xs text-gray-400 mb-2">
                      配置用于生成训练数据的问答模板。支持占位符：{'{organ}'} 表示器官名称
                    </div>

                    {/* 图例说明 */}
                    <div className="flex items-center gap-4 text-xs text-gray-500 mb-2">
                      <span className="flex items-center gap-1">
                        <div className="w-3 h-3 rounded border border-green-500 bg-green-500/20" /> 启用
                      </span>
                      <span className="flex items-center gap-1">
                        <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" /> 重点优先
                      </span>
                    </div>

                    {/* 模板类别列表 - 显示所有类别 */}
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {Object.entries(qaTemplates.categories).map(([category, data]) => (
                        <div
                          key={category}
                          className={`border rounded p-2 transition-colors ${
                            data.enabled
                              ? data.priority
                                ? 'border-yellow-500/50 bg-yellow-900/10'
                                : 'border-green-600/50 bg-green-900/10'
                              : 'border-gray-700 bg-gray-800/50 opacity-60'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-2">
                              {/* 启用 checkbox */}
                              <button
                                onClick={async () => {
                                  try {
                                    await setTemplateSelection({ category, enabled: !data.enabled });
                                    const updated = await getQATemplates();
                                    setQaTemplates(updated);
                                  } catch (error) {
                                    console.error('更新模板状态失败:', error);
                                  }
                                }}
                                className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                                  data.enabled
                                    ? 'border-green-500 bg-green-500 text-white'
                                    : 'border-gray-500 bg-transparent'
                                }`}
                                title={data.enabled ? '点击禁用' : '点击启用'}
                              >
                                {data.enabled && <Check className="w-3 h-3" />}
                              </button>

                              {/* 重点星标 */}
                              <button
                                onClick={async () => {
                                  try {
                                    await setTemplateSelection({ category, priority: !data.priority });
                                    const updated = await getQATemplates();
                                    setQaTemplates(updated);
                                  } catch (error) {
                                    console.error('更新模板优先级失败:', error);
                                  }
                                }}
                                className={`p-0.5 transition-colors ${
                                  data.priority
                                    ? 'text-yellow-400'
                                    : 'text-gray-500 hover:text-yellow-400/50'
                                }`}
                                title={data.priority ? '取消重点' : '标记为重点（优先使用）'}
                              >
                                <Star className={`w-4 h-4 ${data.priority ? 'fill-yellow-400' : ''}`} />
                              </button>

                              <span className={`text-sm ${data.enabled ? 'text-gray-300' : 'text-gray-500'}`}>
                                {qaTemplates.description[category] || category}
                              </span>
                              {data.is_custom && (
                                <span className="text-xs px-1.5 py-0.5 bg-purple-900/50 text-purple-300 rounded">自定义</span>
                              )}
                            </div>
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => {
                                  setEditingCategory(category);
                                  setEditingTemplates([...data.templates]);
                                }}
                                className="text-xs text-blue-400 hover:text-blue-300 px-2 py-0.5"
                              >
                                编辑
                              </button>
                              {data.is_custom && (
                                <button
                                  onClick={async () => {
                                    await resetQATemplates(category);
                                    const updated = await getQATemplates();
                                    setQaTemplates(updated);
                                  }}
                                  className="text-xs text-gray-400 hover:text-gray-300 px-2 py-0.5"
                                  title="重置为默认"
                                >
                                  <RotateCcw className="w-3 h-3" />
                                </button>
                              )}
                            </div>
                          </div>
                          <div className="text-xs text-gray-500 ml-12">
                            {data.templates.slice(0, 2).map((t, i) => (
                              <div key={i} className="truncate">• {t}</div>
                            ))}
                            {data.templates.length > 2 && (
                              <div className="text-gray-600">...还有 {data.templates.length - 2} 条</div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* 模板编辑弹窗 */}
                    {editingCategory && (
                      <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-[60]">
                        <div className="bg-gray-800 rounded-lg w-full max-w-lg max-h-[80vh] overflow-hidden flex flex-col">
                          <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                            <h4 className="text-sm font-medium text-white">
                              编辑模板: {qaTemplates.description[editingCategory] || editingCategory}
                            </h4>
                            <button onClick={() => setEditingCategory(null)} className="text-gray-400 hover:text-white">
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                          <div className="flex-1 overflow-y-auto p-4 space-y-2">
                            {editingTemplates.map((template, index) => (
                              <div key={index} className="flex items-center gap-2">
                                <input
                                  type="text"
                                  value={template}
                                  onChange={(e) => {
                                    const newTemplates = [...editingTemplates];
                                    newTemplates[index] = e.target.value;
                                    setEditingTemplates(newTemplates);
                                  }}
                                  className="flex-1 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                                  placeholder="输入问题模板..."
                                />
                                <button
                                  onClick={() => {
                                    const newTemplates = editingTemplates.filter((_, i) => i !== index);
                                    setEditingTemplates(newTemplates);
                                  }}
                                  className="text-red-400 hover:text-red-300 p-1"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </div>
                            ))}
                            <button
                              onClick={() => setEditingTemplates([...editingTemplates, ''])}
                              className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 mt-2"
                            >
                              <Plus className="w-3 h-3" /> 添加模板
                            </button>
                          </div>
                          <div className="p-4 border-t border-gray-700 flex justify-end gap-2">
                            <button
                              onClick={() => setEditingCategory(null)}
                              className="px-3 py-1.5 bg-gray-600 hover:bg-gray-500 rounded text-sm text-white"
                            >
                              取消
                            </button>
                            <button
                              onClick={async () => {
                                const validTemplates = editingTemplates.filter(t => t.trim());
                                if (validTemplates.length === 0) {
                                  alert('至少需要一个模板');
                                  return;
                                }
                                try {
                                  await setQATemplates({ [editingCategory]: validTemplates });
                                  const updated = await getQATemplates();
                                  setQaTemplates(updated);
                                  setEditingCategory(null);
                                } catch (error) {
                                  console.error('保存模板失败:', error);
                                  alert('保存失败');
                                }
                              }}
                              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded text-sm text-white"
                            >
                              保存
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 加载状态提示 */}
              {previewLoading && (
                <div className="mt-4 bg-blue-900/30 border border-blue-500/50 rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <Loader2 className="w-5 h-5 text-blue-400 animate-spin shrink-0" />
                    <div className="flex-1">
                      <h4 className="text-blue-400 font-medium text-sm">
                        {exportConfig.use_llm ? '正在调用大模型生成问答...' : '正在生成预览数据...'}
                      </h4>
                      <p className="text-xs text-gray-400 mt-1">
                        {exportConfig.use_llm
                          ? '大模型生成需要较长时间（约10-30秒/切片），请耐心等待。可在后端控制台查看详细进度。'
                          : '使用模板生成，预计几秒内完成。'}
                      </p>
                      {exportConfig.use_llm && (
                        <div className="mt-2 text-xs text-gray-500">
                          💡 提示：如遇超时，可尝试增大切片步长以减少处理数量
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* 错误提示区域 */}
              {previewError && !previewLoading && (
                <div className="mt-4 bg-red-900/30 border border-red-500/50 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="shrink-0 mt-0.5">
                      <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="text-red-400 font-medium text-sm">{previewError.message}</h4>
                        {previewError.code && (
                          <span className="text-xs px-1.5 py-0.5 bg-red-800/50 text-red-300 rounded">{previewError.code}</span>
                        )}
                      </div>
                      {previewError.details && (
                        <pre className="text-xs text-gray-400 whitespace-pre-wrap mt-2 p-2 bg-gray-900/50 rounded overflow-auto max-h-32">
                          {previewError.details}
                        </pre>
                      )}
                      <div className="mt-3 flex gap-2">
                        <button
                          onClick={() => setPreviewError(null)}
                          className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
                        >
                          关闭提示
                        </button>
                        <button
                          onClick={handleGeneratePreview}
                          disabled={previewLoading}
                          className="text-xs px-2 py-1 bg-purple-600 hover:bg-purple-700 rounded text-white flex items-center gap-1"
                        >
                          <RotateCcw className="w-3 h-3" /> 重试
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 步骤2: 预览审核 */}
          {currentStep === 'preview' && previewData && (
            <div className="h-full flex flex-col">
              {currentSliceGroup && (
                <div className="flex-1 flex gap-4 min-h-0">
                  {/* 左侧：双图展示区（上下布局） */}
                  <div className="w-[520px] shrink-0 flex flex-col gap-2">
                    {/* 原始CT图像 */}
                    <div className="flex-1 bg-black rounded-lg overflow-hidden flex flex-col min-h-0">
                      <div className="px-2 py-1 bg-gray-800 text-xs text-gray-400 flex items-center justify-between">
                        <span>原始 CT</span>
                        <span>切片 {currentSliceGroup.slice_idx} | {currentSliceGroup.axis}</span>
                      </div>
                      <div className="flex-1 min-h-0">
                        <ZoomableImage
                          src={`data:image/png;base64,${currentSliceGroup.image_base64}`}
                          alt="原始CT"
                          showControls={true}
                        />
                      </div>
                    </div>

                    {/* 分割叠加图像 */}
                    <div className="flex-1 bg-black rounded-lg overflow-hidden flex flex-col min-h-0">
                      <div className="px-2 py-1 bg-gray-800 text-xs text-gray-400 flex items-center justify-between">
                        <span>分割叠加</span>
                        <span>{currentSliceGroup.organs.length} 个器官</span>
                      </div>
                      <div className="flex-1 min-h-0">
                        <ZoomableImage
                          src={`data:image/png;base64,${currentSliceGroup.seg_overlay_base64 || currentSliceGroup.image_base64}`}
                          alt="分割叠加"
                          showControls={true}
                        />
                      </div>
                    </div>

                    {/* 器官标签 */}
                    <div className="bg-gray-800 rounded-lg p-2 shrink-0">
                      <div className="text-xs text-gray-500 mb-1">检测器官 ({currentSliceGroup.organs.length})</div>
                      <div className="flex flex-wrap gap-1 max-h-16 overflow-y-auto">
                        {currentSliceGroup.organs.map(organ => (
                          <span key={organ} className="px-1.5 py-0.5 bg-blue-900/50 text-blue-300 rounded text-xs" title={organ}>{getOrganDisplayName(organ)}</span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* 右侧：问答内容列表 */}
                  <div className="flex-1 flex flex-col min-h-0 bg-gray-800 rounded-lg overflow-hidden">
                    {/* 统计信息栏 */}
                    <div className="px-3 py-2 border-b border-gray-700 flex items-center justify-between shrink-0">
                      <div className="text-sm text-gray-400">
                        该切片 {currentSliceGroup.qa_pairs.length} 个问答对
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-400">已选: {selectedItems.size} / {previewData.preview.length}</span>
                        <button onClick={() => setSelectedItems(new Set(previewData.preview.map(i => i.id)))} className="text-xs text-blue-400 hover:text-blue-300">全选</button>
                        <button onClick={() => setSelectedItems(new Set())} className="text-xs text-gray-400 hover:text-gray-300">清空</button>
                      </div>
                    </div>

                    {/* 问答对列表 */}
                    <div className="flex-1 overflow-y-auto p-3 space-y-3">
                      {currentSliceGroup.qa_pairs.map((qa, qaIndex) => (
                        <div
                          key={qa.id}
                          className={`border rounded-lg p-3 ${
                            selectedItems.has(qa.id)
                              ? 'border-green-600 bg-green-900/20'
                              : 'border-gray-700 bg-gray-700/30'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-gray-500">问答 #{qaIndex + 1}</span>
                              {qa.source === 'manual_edit' && (
                                <span className="text-xs text-yellow-500">(已编辑)</span>
                              )}
                              {qa.source === 'llm' && (
                                <span className="text-xs text-blue-400">(LLM)</span>
                              )}
                            </div>
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => startEditQA(qa)}
                                className="px-2 py-0.5 rounded text-xs bg-blue-600 hover:bg-blue-500 text-white"
                                title="编辑问答"
                              >
                                编辑
                              </button>
                              <button
                                onClick={() => toggleItemSelection(qa.id)}
                                className={`px-2 py-0.5 rounded text-xs flex items-center gap-1 ${
                                  selectedItems.has(qa.id)
                                    ? 'bg-green-600 text-white'
                                    : 'bg-gray-600 text-gray-300'
                                }`}
                              >
                                {selectedItems.has(qa.id) ? <><Check className="w-3 h-3" />选中</> : <><X className="w-3 h-3" />未选</>}
                              </button>
                            </div>
                          </div>
                          {exportConfig.format === 'sharegpt' ? (
                            <>
                              <div className="mb-2">
                                <div className="text-xs text-purple-400 mb-1">Q:</div>
                                <div className="text-sm text-gray-300 bg-gray-800/50 p-2 rounded">{qa.question}</div>
                              </div>
                              <div>
                                <div className="text-xs text-green-400 mb-1">A:</div>
                                <div className="text-sm text-gray-300 bg-gray-800/50 p-2 rounded max-h-32 overflow-y-auto">{qa.answer}</div>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="mb-2">
                                <div className="text-xs text-purple-400 mb-1">指令:</div>
                                <div className="text-sm text-gray-300 bg-gray-800/50 p-2 rounded">{qa.instruction}</div>
                              </div>
                              <div>
                                <div className="text-xs text-green-400 mb-1">输出:</div>
                                <div className="text-sm text-gray-300 bg-gray-800/50 p-2 rounded max-h-32 overflow-y-auto">{qa.output}</div>
                              </div>
                            </>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 步骤3: 导出完成 */}
          {currentStep === 'export' && exportResult && (
            <div className="space-y-4">
              <div className="text-center py-8">
                <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">导出完成！</h3>
                <p className="text-gray-400">训练数据已成功导出并下载</p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 space-y-2">
                <div className="flex justify-between text-sm"><span className="text-gray-400">总样本数</span><span className="text-white">{exportResult.total_samples}</span></div>
                <div className="flex justify-between text-sm"><span className="text-gray-400">数据格式</span><span className="text-white">{exportResult.format}</span></div>
                <div className="flex justify-between text-sm"><span className="text-gray-400">使用 LLM</span><span className="text-white">{exportResult.config?.use_llm ? '是' : '否（模板生成）'}</span></div>
                <div className="flex justify-between text-sm"><span className="text-gray-400">数据来源</span><span className="text-white">{exportResult.config?.exported_from_preview ? '预览数据筛选' : '实时生成'}</span></div>
              </div>
            </div>
          )}
        </div>

        {/* 底部操作按钮 */}
        <div className="flex items-center justify-between gap-2 px-4 py-2 border-t border-gray-700">
          {/* 左侧：重新开始按钮 */}
          <div className="flex items-center gap-2">
            {currentStep !== 'config' && (
              <button onClick={handleReset} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm text-white">
                重新开始
              </button>
            )}
          </div>

          {/* 中间：切片导航（仅预览阶段显示） */}
          {currentStep === 'preview' && groupedPreviewData.length > 0 && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPreviewIndex(i => Math.max(0, i - 1))}
                disabled={currentPreviewIndex === 0}
                className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>

              {/* 切片选择器 */}
              <div className="flex items-center gap-1">
                <span className="text-sm text-gray-400">切片</span>
                <select
                  value={currentPreviewIndex}
                  onChange={(e) => setCurrentPreviewIndex(Number(e.target.value))}
                  className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm text-white w-20"
                >
                  {groupedPreviewData.map((group, idx) => (
                    <option key={`${group.slice_idx}-${group.axis}`} value={idx}>
                      {idx + 1} / {groupedPreviewData.length}
                    </option>
                  ))}
                </select>
                <span className="text-xs text-gray-500">({currentSliceGroup?.slice_idx})</span>
              </div>

              <button
                onClick={() => setCurrentPreviewIndex(i => Math.min(groupedPreviewData.length - 1, i + 1))}
                disabled={currentPreviewIndex === groupedPreviewData.length - 1}
                className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>

              {/* 快速跳转按钮 */}
              <div className="flex items-center gap-0.5 ml-2">
                {[0, Math.floor(groupedPreviewData.length / 4), Math.floor(groupedPreviewData.length / 2), Math.floor(groupedPreviewData.length * 3 / 4), groupedPreviewData.length - 1]
                  .filter((v, i, a) => a.indexOf(v) === i) // 去重
                  .map((idx) => (
                    <button
                      key={idx}
                      onClick={() => setCurrentPreviewIndex(idx)}
                      className={`w-6 h-6 rounded text-xs ${
                        idx === currentPreviewIndex
                          ? 'bg-purple-600 text-white'
                          : groupedPreviewData[idx]?.qa_pairs.every(qa => selectedItems.has(qa.id))
                            ? 'bg-green-700 text-white'
                            : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                      }`}
                      title={`切片 ${groupedPreviewData[idx]?.slice_idx}`}
                    >
                      {idx + 1}
                    </button>
                  ))}
              </div>
            </div>
          )}

          {/* 右侧：操作按钮 */}
          <div className="flex gap-2">
            <button onClick={onClose} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm text-white">关闭</button>
            {currentStep === 'config' && (
              <button onClick={handleGeneratePreview} disabled={!sessionId || previewLoading} className="px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded text-sm text-white flex items-center gap-2 disabled:opacity-50">
                {previewLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
                {previewLoading ? '生成中...' : '生成预览'}
              </button>
            )}
            {currentStep === 'preview' && (
              <button onClick={handleExport} disabled={!sessionId || exporting || selectedItems.size === 0} className="px-3 py-1.5 bg-green-600 hover:bg-green-700 rounded text-sm text-white flex items-center gap-2 disabled:opacity-50">
                {exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                {exporting ? '导出中...' : `导出 ${selectedItems.size} 条`}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* 问答编辑弹窗 - 带图像展示 */}
      {editingQAId && (() => {
        // 找到正在编辑的问答项
        const editingItem = previewData?.preview.find(item => item.id === editingQAId);
        return (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-70">
            <div className="bg-gray-800 rounded-lg w-[98vw] max-w-[1600px] h-[95vh] overflow-hidden flex flex-col">
              <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
                <h4 className="text-sm font-medium text-white">
                  编辑问答对 - 切片 {editingItem?.slice_idx} | {editingItem?.axis} | {editingItem?.organs.length} 个器官
                </h4>
                <button onClick={cancelEditQA} className="text-gray-400 hover:text-white">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex-1 flex min-h-0 overflow-hidden">
                {/* 左侧：双图展示 */}
                <div className="w-[800px] shrink-0 flex flex-col gap-2 p-3 bg-gray-900">
                  {/* 原始CT图像 */}
                  <div className="flex-1 bg-black rounded-lg overflow-hidden flex flex-col min-h-0">
                    <div className="px-2 py-1 bg-gray-800 text-xs text-gray-400">原始 CT</div>
                    <div className="flex-1 min-h-0">
                      {editingItem && (
                        <ZoomableImage
                          src={`data:image/png;base64,${editingItem.image_base64}`}
                          alt="原始CT"
                          showControls={true}
                        />
                      )}
                    </div>
                  </div>

                  {/* 分割叠加图像 */}
                  <div className="flex-1 bg-black rounded-lg overflow-hidden flex flex-col min-h-0">
                    <div className="px-2 py-1 bg-gray-800 text-xs text-gray-400">分割叠加</div>
                    <div className="flex-1 min-h-0">
                      {editingItem && (
                        <ZoomableImage
                          src={`data:image/png;base64,${editingItem.seg_overlay_base64 || editingItem.image_base64}`}
                          alt="分割叠加"
                          showControls={true}
                        />
                      )}
                    </div>
                  </div>

                  {/* 器官列表 */}
                  {editingItem && (
                    <div className="bg-gray-800 rounded-lg p-2 shrink-0">
                      <div className="text-xs text-gray-500 mb-1">检测器官 ({editingItem.organs.length})</div>
                      <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                        {editingItem.organs.map(organ => (
                          <span key={organ} className="px-1.5 py-0.5 bg-blue-900/50 text-blue-300 rounded text-xs" title={organ}>{getOrganDisplayName(organ)}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* 右侧：问答编辑区 */}
                <div className="flex-1 flex flex-col min-h-0 p-5">
                  <div className="flex-1 flex flex-col gap-5 overflow-y-auto">
                    <div className="shrink-0">
                      <label className="block text-sm text-gray-400 mb-2">问题 (Q):</label>
                      <textarea
                        value={editingQuestion}
                        onChange={(e) => setEditingQuestion(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-600 rounded p-3 text-white text-sm resize-none focus:border-purple-500 focus:outline-none"
                        rows={5}
                        placeholder="输入问题..."
                      />
                    </div>
                    <div className="flex-1 flex flex-col min-h-0">
                      <label className="block text-sm text-gray-400 mb-2">答案 (A):</label>
                      <textarea
                        value={editingAnswer}
                        onChange={(e) => setEditingAnswer(e.target.value)}
                        className="flex-1 w-full bg-gray-900 border border-gray-600 rounded p-3 text-white text-sm resize-none focus:border-green-500 focus:outline-none min-h-[200px]"
                        placeholder="输入答案..."
                      />
                    </div>
                  </div>

                  {/* 底部按钮 */}
                  <div className="flex justify-end gap-2 pt-4 shrink-0">
                    <button
                      onClick={cancelEditQA}
                      className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded text-sm text-white"
                    >
                      取消
                    </button>
                    <button
                      onClick={saveEditQA}
                      className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded text-sm text-white"
                    >
                      保存修改
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

