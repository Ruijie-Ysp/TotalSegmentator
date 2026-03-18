'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Database, 
  Upload, 
  CheckCircle, 
  XCircle, 
  Loader2, 
  ExternalLink,
  Settings,
  RefreshCw
} from 'lucide-react';
import { 
  getLabelStudioStatus, 
  syncToLabelStudio, 
  LabelStudioStatus, 
  LabelStudioSyncResult 
} from '@/lib/api';

interface LabelStudioSyncProps {
  sessionId: string;
  filename?: string;
  selectedOrgans?: string[];
  onSyncComplete?: (result: LabelStudioSyncResult) => void;
}

export default function LabelStudioSync({ sessionId, filename, selectedOrgans = [], onSyncComplete }: LabelStudioSyncProps) {
  const [status, setStatus] = useState<LabelStudioStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<LabelStudioSyncResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 同步配置
  const [sliceStep, setSliceStep] = useState(10);
  const [axis, setAxis] = useState<'axial' | 'sagittal' | 'coronal'>('axial');
  const [projectName, setProjectName] = useState('');

  // 根据文件名生成默认项目名称
  useEffect(() => {
    if (filename) {
      // 移除文件扩展名
      const baseName = filename.replace(/\.(nii\.gz|nii|zip|dcm)$/i, '');
      setProjectName(`影像分割-${baseName}`);
    } else {
      setProjectName('影像分割-未知文件');
    }
  }, [filename]);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getLabelStudioStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError('无法获取 Label Studio 状态');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleSync = async () => {
    setSyncing(true);
    setSyncResult(null);
    setError(null);

    try {
      const result = await syncToLabelStudio(sessionId, {
        projectName,
        sliceStep,
        axis,
        organs: selectedOrgans
      });
      setSyncResult(result);
      onSyncComplete?.(result);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : '同步失败';
      setError(errorMessage);
    } finally {
      setSyncing(false);
    }
  };

  const isReady = status?.labelstudio.connected && status?.minio.connected;

  return (
    <div className="p-2 bg-gray-900 rounded-lg border border-gray-700 text-xs space-y-2">
      {/* 标题栏 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-gray-300 font-medium">
          <Database className="w-3.5 h-3.5" />
          <span>标注系统同步</span>
        </div>
        <button
          onClick={fetchStatus}
          disabled={loading}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
          title="刷新状态"
        >
          <RefreshCw className={`w-3 h-3 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* 状态指示器 */}
      <div className="grid grid-cols-2 gap-1.5">
        <div className="flex items-center gap-1.5 px-2 py-1 bg-gray-800 rounded">
          {status?.labelstudio.connected ? (
            <CheckCircle className="w-3 h-3 text-green-400" />
          ) : (
            <XCircle className="w-3 h-3 text-red-400" />
          )}
          <span className="text-gray-400">Label Studio</span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-1 bg-gray-800 rounded">
          {status?.minio.connected ? (
            <CheckCircle className="w-3 h-3 text-green-400" />
          ) : (
            <XCircle className="w-3 h-3 text-red-400" />
          )}
          <span className="text-gray-400">MinIO</span>
        </div>
      </div>

      {/* 配置不完整提示 */}
      {status && !status.labelstudio.configured && (
        <div className="p-1.5 bg-amber-900/30 border border-amber-600/50 rounded text-amber-300">
          ⚠️ API Key 未配置
        </div>
      )}

      {status && !status.labelstudio.sdk_available && (
        <div className="p-1.5 bg-amber-900/30 border border-amber-600/50 rounded text-amber-300">
          ⚠️ SDK 未安装
        </div>
      )}

      {/* 同步配置 */}
      {isReady && (
        <div className="space-y-2 border-t border-gray-700 pt-2">
          <div className="flex items-center gap-1.5 text-gray-400">
            <Settings className="w-3 h-3" />
            <span>同步配置</span>
          </div>

          <div className="grid grid-cols-2 gap-1.5">
            <div>
              <label className="block text-gray-500 mb-0.5">切片间隔</label>
              <select
                value={sliceStep}
                onChange={(e) => setSliceStep(Number(e.target.value))}
                className="w-full bg-gray-800 border border-gray-600 text-gray-200 rounded px-2 py-1"
              >
                <option value={5}>每 5 层</option>
                <option value={10}>每 10 层</option>
                <option value={20}>每 20 层</option>
                <option value={50}>每 50 层</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-500 mb-0.5">切片方向</label>
              <select
                value={axis}
                onChange={(e) => setAxis(e.target.value as 'axial' | 'sagittal' | 'coronal')}
                className="w-full bg-gray-800 border border-gray-600 text-gray-200 rounded px-2 py-1"
              >
                <option value="axial">轴向 (Ax)</option>
                <option value="sagittal">矢状 (Sag)</option>
                <option value="coronal">冠状 (Cor)</option>
              </select>
            </div>
          </div>

          {/* 项目名称（可编辑） */}
          <div>
            <label className="block text-gray-500 mb-0.5">项目名称</label>
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 text-gray-200 rounded px-2 py-1"
              placeholder="Label Studio 项目名称"
            />
          </div>
        </div>
      )}

      {/* 同步按钮 */}
      <button
        onClick={handleSync}
        disabled={!isReady || syncing}
        className={`w-full flex items-center justify-center gap-1.5 py-1.5 rounded font-medium transition-colors ${
          isReady && !syncing
            ? 'bg-blue-600 hover:bg-blue-700 text-white'
            : 'bg-gray-700 text-gray-400 cursor-not-allowed'
        }`}
      >
        {syncing ? (
          <>
            <Loader2 className="w-3 h-3 animate-spin" />
            同步中...
          </>
        ) : (
          <>
            <Upload className="w-3 h-3" />
            同步到标注系统
          </>
        )}
      </button>

      {/* 同步结果 */}
      {syncResult && (
        <div className={`p-1.5 rounded ${
          syncResult.success ? 'bg-green-900/30 border border-green-600/50' : 'bg-red-900/30 border border-red-600/50'
        }`}>
          <p className={syncResult.success ? 'text-green-300' : 'text-red-300'}>
            {syncResult.success ? '✓' : '✗'} {syncResult.message}
          </p>
          {syncResult.success && (
            <a
              href={`${status?.labelstudio.url}/projects/${syncResult.project_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 flex items-center gap-1 mt-1"
            >
              打开项目 <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>
      )}

      {/* 错误信息 */}
      {error && (
        <div className="p-1.5 bg-red-900/30 border border-red-600/50 rounded text-red-300">
          {error}
        </div>
      )}

      {/* 使用说明 */}
      {!isReady && status && (
        <div className="text-gray-500 space-y-0.5">
          <p>配置:</p>
          <ol className="list-decimal list-inside pl-1">
            <li>启动 Label Studio</li>
            <li>创建 API Key</li>
            <li>配置 .env</li>
          </ol>
        </div>
      )}
    </div>
  );
}

