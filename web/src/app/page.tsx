'use client';

import React, { useState, useEffect, useCallback, useRef, lazy, Suspense } from 'react';
import FileUploader from '@/components/FileUploader';
import SliceViewer from '@/components/SliceViewer';
import OrganSelector from '@/components/OrganSelector';
import ControlPanel from '@/components/ControlPanel';
import TrainingExportPanel, { ActivityStatus } from '@/components/TrainingExportPanel';
import TaskList from '@/components/TaskList';
import LabelStudioSync from '@/components/LabelStudioSync';
import {
  uploadFile,
  runSegmentation,
  getSliceInfo,
  getWindowPresets,
  getTasks,
  getSystemInfo,
  getExportNiftiUrl,
  exportLabelStudioSimple,
  getMetadata,
  getActiveTask,
  getTaskStatus,
  SliceInfo,
  WindowPreset,
  SessionInfo,
  SystemInfo,
  OverlayMode,
} from '@/lib/api';
import { Activity, Brain, Loader2 } from 'lucide-react';

// 动态导入 NiiVue 组件（避免 SSR 问题）
const NiiVueViewer = lazy(() => import('@/components/NiiVueViewer'));

type Status = 'idle' | 'uploading' | 'processing' | 'completed' | 'error' | 'restoring';

// 会话活动状态类型（用于任务列表显示）
export interface SessionActivity {
  sessionId: string;
  activity: ActivityStatus;
}

// localStorage 键名
const STORAGE_KEY = 'totalseg_session';

interface StoredSession {
  sessionId: string;
  sessionInfo: SessionInfo;
  sliceInfo: SliceInfo;
  selectedOrgans: string[];
  windowCenter: number;
  windowWidth: number;
  overlayOpacity: number;
  overlayMode: OverlayMode;
  contourThickness: number;
  task: string;
  savedAt: number;
}

// 保存会话到 localStorage
function saveSession(data: StoredSession) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (e) {
    console.error('Failed to save session:', e);
  }
}

// 从 localStorage 加载会话
function loadSession(): StoredSession | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const data = JSON.parse(stored) as StoredSession;
      // 检查是否在24小时内
      const age = Date.now() - data.savedAt;
      if (age < 24 * 60 * 60 * 1000) {
        return data;
      }
    }
  } catch (e) {
    console.error('Failed to load session:', e);
  }
  return null;
}

// 清除存储的会话
function clearSession() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    console.error('Failed to clear session:', e);
  }
}

export default function Home() {
  // 会话状态
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);

  // 视图参数
  const [sliceInfo, setSliceInfo] = useState<SliceInfo | null>(null);
  const [windowCenter, setWindowCenter] = useState(40);
  const [windowWidth, setWindowWidth] = useState(400);
  const [overlayOpacity, setOverlayOpacity] = useState(1);  // 默认 100%
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('fill_contour');  // 默认填充+边界
  const [contourThickness, setContourThickness] = useState(1);  // 默认 1px
  const [selectedOrgans, setSelectedOrgans] = useState<string[]>([]);

  // 切片位置状态（用于 2D/3D 同步）
  const [slicePositions, setSlicePositions] = useState({
    axial: 0,
    sagittal: 0,
    coronal: 0,
  });

  // 切片变化回调
  const handleSliceChange = useCallback((axis: 'axial' | 'sagittal' | 'coronal', sliceIdx: number) => {
    setSlicePositions(prev => ({
      ...prev,
      [axis]: sliceIdx,
    }));
  }, []);

  // 配置
  const [windowPresets, setWindowPresets] = useState<Record<string, WindowPreset>>({});
  const [availableTasks, setAvailableTasks] = useState<string[]>(['total']);
  const [task, setTask] = useState('total');
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  // 训练数据导出面板
  const [showTrainingExport, setShowTrainingExport] = useState(false);

  // 会话活动状态（用于任务列表显示生成预览/导出中状态）
  const [sessionActivity, setSessionActivity] = useState<SessionActivity | null>(null);

  // 处理活动状态变化
  const handleActivityChange = useCallback((activity: ActivityStatus, activitySessionId: string | null) => {
    if (activitySessionId && activity !== 'idle') {
      setSessionActivity({ sessionId: activitySessionId, activity });
    } else {
      setSessionActivity(null);
    }
  }, []);

  // 加载配置和恢复会话
  useEffect(() => {
    getWindowPresets().then(setWindowPresets).catch(console.error);
    getTasks()
      .then((tasks) => setAvailableTasks(Object.keys(tasks)))
      .catch(console.error);
    getSystemInfo().then(setSystemInfo).catch(console.error);

    // 首先检查是否有进行中的任务
    getActiveTask()
      .then(async (result) => {
        if (result.active_task) {
          // 有进行中的任务，恢复并开始轮询
          const activeTask = result.active_task;
          setSessionId(activeTask.session_id);
          setStatus('processing');
          setStatusMessage(`正在恢复任务: ${activeTask.filename} (${activeTask.progress}%)`);
          return; // 让轮询 effect 处理后续
        }

        // 没有进行中的任务，尝试恢复之前完成的会话
        const stored = loadSession();
        if (stored) {
          setStatus('restoring');
          setStatusMessage('正在恢复上次会话...');

          try {
            const meta = await getMetadata(stored.sessionId);
            if (meta.status === 'completed' && meta.data_available !== false) {
              // 恢复会话状态
              setSessionId(stored.sessionId);
              setSessionInfo(stored.sessionInfo);
              setSliceInfo(stored.sliceInfo);
              setSelectedOrgans(stored.selectedOrgans);
              setWindowCenter(stored.windowCenter);
              setWindowWidth(stored.windowWidth);
              setOverlayOpacity(stored.overlayOpacity);
              setOverlayMode(stored.overlayMode);
              setContourThickness(stored.contourThickness);
              setTask(stored.task);
              setStatus('completed');
              setStatusMessage('会话已恢复');
            } else {
              // 会话已过期或数据不可用
              clearSession();
              setStatus('idle');
              setStatusMessage('');
            }
          } catch {
            // 后端会话不存在，清除本地存储
            clearSession();
            setStatus('idle');
            setStatusMessage('');
          }
        }
      })
      .catch(console.error);
  }, []);

  // 轮询进行中任务的状态
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // 只在 processing 状态且有 sessionId 时轮询
    if (status !== 'processing' || !sessionId) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    const pollStatus = async () => {
      try {
        const taskStatus = await getTaskStatus(sessionId);

        if (taskStatus.status === 'completed') {
          // 任务完成，加载完整数据
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }

          const sliceData = await getSliceInfo(sessionId);
          setSliceInfo(sliceData);

          const newSessionInfo: SessionInfo = {
            session_id: sessionId,
            filename: taskStatus.filename || '',
            status: 'completed',
            organs: taskStatus.organs || [],
            shape: taskStatus.shape,
            device_used: taskStatus.device_used,
            task: taskStatus.task_type,
          };
          setSessionInfo(newSessionInfo);

          // 默认全选所有识别出的器官
          const newSelectedOrgans = taskStatus.organs || [];
          setSelectedOrgans(newSelectedOrgans);

          setStatus('completed');
          setStatusMessage(`处理完成！(设备: ${taskStatus.device_used || 'unknown'})`);

          // 保存会话到 localStorage
          saveSession({
            sessionId,
            sessionInfo: newSessionInfo,
            sliceInfo: sliceData,
            selectedOrgans: newSelectedOrgans,
            windowCenter: 40,
            windowWidth: 400,
            overlayOpacity: 1,
            overlayMode: 'fill_contour',
            contourThickness: 1,
            task: taskStatus.task_type || task,
            savedAt: Date.now(),
          });
        } else if (taskStatus.status === 'error') {
          // 任务出错
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
          setStatus('error');
          setStatusMessage(`处理失败: ${taskStatus.error_message || '未知错误'}`);
        } else {
          // 仍在处理中，更新进度
          setStatusMessage(`${taskStatus.progress_message || '处理中...'} (${taskStatus.progress}%)`);
        }
      } catch (err) {
        console.error('轮询任务状态失败:', err);
      }
    };

    // 立即执行一次
    pollStatus();

    // 每 2 秒轮询一次
    pollingRef.current = setInterval(pollStatus, 2000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [status, sessionId, task]);

  // 上传和处理文件
  const handleUpload = useCallback(async (file: File) => {
    try {
      // 清除旧会话
      clearSession();

      setStatus('uploading');
      setStatusMessage(`正在上传 ${file.name}...`);

      const result = await uploadFile(file);
      setSessionId(result.session_id);

      setStatus('processing');
      const deviceHint = systemInfo?.device.recommended_device || 'auto';
      setStatusMessage(`正在运行分割模型 (设备: ${deviceHint})...`);

      // 使用自动设备检测
      const segResult = await runSegmentation(result.session_id, task, true, 'auto');
      setSessionInfo(segResult);

      const sliceData = await getSliceInfo(result.session_id);
      setSliceInfo(sliceData);

      // 默认全选所有识别出的器官
      const newSelectedOrgans = segResult.organs || [];
      setSelectedOrgans(newSelectedOrgans);

      setStatus('completed');
      const usedDevice = segResult.device_used || 'unknown';
      setStatusMessage(`处理完成！(设备: ${usedDevice})`);

      // 保存会话到 localStorage
      saveSession({
        sessionId: result.session_id,
        sessionInfo: segResult,
        sliceInfo: sliceData,
        selectedOrgans: newSelectedOrgans,
        windowCenter: 40,
        windowWidth: 400,
        overlayOpacity: 1,  // 默认 100%
        overlayMode: 'fill_contour',  // 默认填充+边界
        contourThickness: 1,  // 默认 1px
        task,
        savedAt: Date.now(),
      });
    } catch (error) {
      console.error(error);
      setStatus('error');
      setStatusMessage(error instanceof Error ? error.message : '发生未知错误');
    }
  }, [task, systemInfo]);

  // 同步更新 localStorage 中的视图参数
  useEffect(() => {
    if (status === 'completed' && sessionId && sessionInfo && sliceInfo) {
      saveSession({
        sessionId,
        sessionInfo,
        sliceInfo,
        selectedOrgans,
        windowCenter,
        windowWidth,
        overlayOpacity,
        overlayMode,
        contourThickness,
        task,
        savedAt: Date.now(),
      });
    }
  }, [status, sessionId, sessionInfo, sliceInfo, selectedOrgans, windowCenter, windowWidth, overlayOpacity, overlayMode, contourThickness, task]);

  // 导出功能
  const handleExportNifti = useCallback(() => {
    if (sessionId) {
      window.open(getExportNiftiUrl(sessionId), '_blank');
    }
  }, [sessionId]);

  const handleExportLabelStudio = useCallback(async () => {
    if (sessionId) {
      try {
        // 使用简化导出 API - 图片以 Base64 嵌入，直接导入 Label Studio 即可
        const tasks = await exportLabelStudioSimple(sessionId, {
          sliceStep: 20,
          axis: 'axial',
          maxSlices: 30,  // 限制数量避免文件过大
        });

        // 下载 JSON 文件
        const blob = new Blob([JSON.stringify(tasks, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `labelstudio_${sessionId}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (error) {
        console.error('导出失败:', error);
        alert('导出失败，请重试');
      }
    }
  }, [sessionId]);

  const isProcessing = status === 'uploading' || status === 'processing' || status === 'restoring';

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col overflow-hidden">
      {/* 顶部导航 - 极简紧凑设计 */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm shrink-0">
        <div className="px-3 py-1.5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* 返回任务列表按钮 - 仅在分割界面显示 */}
            {status === 'completed' && (
              <button
                onClick={() => {
                  clearSession();
                  setStatus('idle');
                  setStatusMessage('');
                  setSessionId(null);
                  setSessionInfo(null);
                  setSliceInfo(null);
                  setSelectedOrgans([]);
                }}
                className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded transition-colors mr-2"
                title="返回任务列表"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <span className="hidden sm:inline">返回</span>
              </button>
            )}
            <Brain className="w-5 h-5 text-blue-400" />
            <span className="text-sm font-semibold">智能影像分割标注</span>
            {(status === 'processing' || status === 'restoring') && (
              <Activity className="w-4 h-4 text-orange-400 animate-pulse" />
            )}
          </div>
          <div className="flex items-center gap-3 text-xs text-gray-400">
            {systemInfo && (
              <span className="hidden sm:inline">
                {systemInfo.device.recommended_device.toUpperCase()}
              </span>
            )}
            {sessionInfo && (
              <span className="text-green-400">{sessionInfo.organs?.length || 0} 器官</span>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 min-h-0 p-2 overflow-auto">
        {/* 上传区域 - 未完成时显示 */}
        {status !== 'completed' && (
          <div className="flex flex-col gap-6 w-full px-4 lg:px-8 py-8">
            {/* 上传组件 - 居中显示 */}
            <div className="max-w-2xl mx-auto w-full">
              <FileUploader
                onUpload={handleUpload}
                isProcessing={isProcessing}
                status={status}
                statusMessage={statusMessage}
              />
            </div>

            {/* 历史任务列表 - 全宽显示 */}
            <TaskList
              sessionActivity={sessionActivity}
              onTaskSelect={(sid) => {
                // 选择历史任务时，恢复该任务
                setSessionId(sid);
                setStatus('restoring');
                setStatusMessage('正在恢复任务...');
                // 加载任务数据
                Promise.all([
                  getSliceInfo(sid),
                  getMetadata(sid),
                ]).then(([slice, meta]) => {
                  setSliceInfo(slice);
                  setSessionInfo({
                    session_id: sid,
                    filename: meta.filename || 'unknown',
                    file_type: meta.file_type,
                    shape: meta.shape,
                    organs: meta.organs || [],
                    device_used: meta.device_used,
                    status: 'completed',
                  });
                  setSelectedOrgans(meta.organs?.slice(0, 5) || []);
                  setStatus('completed');
                  setStatusMessage('');
                }).catch((e) => {
                  setStatus('error');
                  setStatusMessage(`恢复失败: ${e.message}`);
                });
              }}
            />
          </div>
        )}

        {/* 主视图区域 - 完成后显示 */}
        {status === 'completed' && sessionId && sliceInfo && (
          <div className="h-full flex gap-2">
            {/* 左侧器官选择 */}
            <div className="w-48 shrink-0">
              <OrganSelector
                availableOrgans={sessionInfo?.organs || []}
                selectedOrgans={selectedOrgans}
                onSelectionChange={setSelectedOrgans}
              />
            </div>

            {/* 中间四格视图区域 - 使用绝对定位确保不溢出 */}
            <div className="flex-1 min-w-0 relative">
              <div className="absolute inset-0 grid grid-rows-2 grid-cols-2 gap-2">
                {/* 左上: 轴向视图 */}
                <SliceViewer
                  sessionId={sessionId}
                  axis="axial"
                  maxSlice={sliceInfo.axial.max}
                  defaultSlice={sliceInfo.axial.default}
                  windowCenter={windowCenter}
                  windowWidth={windowWidth}
                  selectedOrgans={selectedOrgans}
                  overlayOpacity={overlayOpacity}
                  overlayMode={overlayMode}
                  contourThickness={contourThickness}
                  onSliceChange={(idx) => handleSliceChange('axial', idx)}
                />
                {/* 右上: 矢状视图 */}
                <SliceViewer
                  sessionId={sessionId}
                  axis="sagittal"
                  maxSlice={sliceInfo.sagittal.max}
                  defaultSlice={sliceInfo.sagittal.default}
                  windowCenter={windowCenter}
                  windowWidth={windowWidth}
                  selectedOrgans={selectedOrgans}
                  overlayOpacity={overlayOpacity}
                  overlayMode={overlayMode}
                  contourThickness={contourThickness}
                  onSliceChange={(idx) => handleSliceChange('sagittal', idx)}
                />
                {/* 左下: 冠状视图 */}
                <SliceViewer
                  sessionId={sessionId}
                  axis="coronal"
                  maxSlice={sliceInfo.coronal.max}
                  defaultSlice={sliceInfo.coronal.default}
                  windowCenter={windowCenter}
                  windowWidth={windowWidth}
                  selectedOrgans={selectedOrgans}
                  overlayOpacity={overlayOpacity}
                  overlayMode={overlayMode}
                  contourThickness={contourThickness}
                  onSliceChange={(idx) => handleSliceChange('coronal', idx)}
                />
                {/* 右下: NiiVue 3D 视图 */}
                <Suspense fallback={
                  <div className="h-full bg-gray-900 rounded-lg border border-gray-700 flex items-center justify-center">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
                  </div>
                }>
                  <NiiVueViewer
                    sessionId={sessionId}
                    selectedOrgans={selectedOrgans}
                    slicePositions={slicePositions}
                    sliceMaxValues={{
                      axial: sliceInfo.axial.max,
                      sagittal: sliceInfo.sagittal.max,
                      coronal: sliceInfo.coronal.max,
                    }}
                  />
                </Suspense>
              </div>
            </div>

            {/* 右侧控制面板 */}
            <div className="w-56 shrink-0 flex flex-col gap-2 overflow-y-auto max-h-full">
              <ControlPanel
                windowCenter={windowCenter}
                windowWidth={windowWidth}
                overlayOpacity={overlayOpacity}
                overlayMode={overlayMode}
                contourThickness={contourThickness}
                windowPresets={windowPresets}
                onWindowChange={(c, w) => { setWindowCenter(c); setWindowWidth(w); }}
                onOpacityChange={setOverlayOpacity}
                onOverlayModeChange={setOverlayMode}
                onContourThicknessChange={setContourThickness}
                onExportNifti={handleExportNifti}
                onExportLabelStudio={handleExportLabelStudio}
                onExportTraining={() => setShowTrainingExport(true)}
                task={task}
                onTaskChange={setTask}
                availableTasks={availableTasks}
                isProcessing={isProcessing}
              />

              {/* 标注系统同步 */}
              <LabelStudioSync
                sessionId={sessionId}
                filename={sessionInfo?.filename}
                selectedOrgans={selectedOrgans}
                onSyncComplete={(result) => {
                  if (result.success) {
                    console.log('同步成功:', result);
                  }
                }}
              />
            </div>
          </div>
        )}
      </main>

      {/* 训练数据导出面板 */}
      <TrainingExportPanel
        sessionId={sessionId}
        isOpen={showTrainingExport}
        onClose={() => setShowTrainingExport(false)}
        onActivityChange={handleActivityChange}
      />
    </div>
  );
}
