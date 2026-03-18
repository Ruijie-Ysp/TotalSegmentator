'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Loader2, RefreshCw, Box, Layers, AlertCircle, Eye, EyeOff, Crosshair } from 'lucide-react';

interface SlicePositions {
  axial: number;
  sagittal: number;
  coronal: number;
}

interface SliceMaxValues {
  axial: number;
  sagittal: number;
  coronal: number;
}

interface NiiVueViewerProps {
  sessionId: string;
  selectedOrgans: string[];
  apiBase?: string;
  slicePositions?: SlicePositions;
  sliceMaxValues?: SliceMaxValues;
}

// 带进度的 fetch
async function fetchWithProgress(
  url: string,
  onProgress: (loaded: number, total: number, stage: string) => void
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`加载失败: ${response.status}`);
  }

  const contentLength = response.headers.get('Content-Length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  if (!response.body) {
    return response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total, '下载中');
  }

  // 合并 chunks
  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  return buffer.buffer;
}

export default function NiiVueViewer({
  sessionId,
  selectedOrgans: _selectedOrgans,
  apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:28000',
  slicePositions,
  sliceMaxValues,
}: NiiVueViewerProps) {
  // 预留 selectedOrgans 供后续使用
  void _selectedOrgans;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'render' | 'multiplanar'>('render');
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [segOpacity, setSegOpacity] = useState(0.5);
  const [progress, setProgress] = useState({ stage: '', percent: 0 });
  const currentSessionRef = useRef<string | null>(null);
  const [showCrosshair, setShowCrosshair] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0); // 刷新计数器

  // 初始化 NiiVue
  useEffect(() => {
    if (currentSessionRef.current === sessionId && nvRef.current) return;
    if (!canvasRef.current) return;

    currentSessionRef.current = sessionId;

    const init = async () => {
      setLoading(true);
      setError(null);
      setProgress({ stage: '初始化', percent: 0 });

      try {
        const { Niivue } = await import('@niivue/niivue');

        if (nvRef.current) {
          try { nvRef.current.closeAllVolumes?.(); } catch { /* ignore */ }
        }

        const nv = new Niivue({
          backColor: [0.05, 0.05, 0.1, 1],
          show3Dcrosshair: showCrosshair,
          crosshairColor: [1, 1, 0, 1],  // 亮黄色不透明十字光标，更容易看清
          crosshairWidth: 3,  // 增加十字光标线宽
        });

        await nv.attachToCanvas(canvasRef.current!);
        nv.setSliceType(nv.sliceTypeRender);
        nvRef.current = nv;

        // 1. 加载 CT 数据
        setProgress({ stage: '下载 CT', percent: 0 });
        const ctBuffer = await fetchWithProgress(
          `${apiBase}/api/volume/ct/${sessionId}`,
          (loaded, total, stage) => {
            const percent = total > 0 ? Math.round((loaded / total) * 100) : 0;
            setProgress({ stage: `${stage} CT`, percent });
          }
        );

        setProgress({ stage: '渲染 CT', percent: 100 });
        await nv.loadFromArrayBuffer(ctBuffer, 'ct.nii.gz');

        // 2. 加载分割数据和颜色映射
        setProgress({ stage: '下载分割', percent: 0 });
        try {
          // 并行获取分割数据和颜色映射
          const [segBuffer, colormapResponse] = await Promise.all([
            fetchWithProgress(
              `${apiBase}/api/volume/seg/${sessionId}`,
              (loaded, total, stage) => {
                const percent = total > 0 ? Math.round((loaded / total) * 100) : 0;
                setProgress({ stage: `${stage} 分割`, percent });
              }
            ),
            fetch(`${apiBase}/api/colormap/${sessionId}`).then(r => r.json())
          ]);

          setProgress({ stage: '渲染分割', percent: 100 });
          // 加载分割作为叠加层
          await nv.loadFromArrayBuffer(segBuffer, 'seg.nii.gz');

          // 设置分割层的显示属性
          if (nv.volumes.length > 1) {
            const segVol = nv.volumes[1];
            // 应用自定义颜色映射（与 2D 切片颜色一致）
            segVol.setColormapLabel(colormapResponse);
            nv.setOpacity(1, segOpacity);
            nv.updateGLVolume();
          }
        } catch (segErr) {
          console.warn('分割数据加载失败，仅显示 CT:', segErr);
        }

        // 设置初始视角
        nv.setRenderAzimuthElevation(110, 15);
        setLoading(false);
      } catch (err) {
        console.error('NiiVue init error:', err);
        setError(`${err instanceof Error ? err.message : '初始化失败'}`);
        setLoading(false);
      }
    };

    init();
  }, [sessionId, apiBase, refreshKey]);

  // 同步 2D 切片位置到 3D 视图
  useEffect(() => {
    if (!nvRef.current || !slicePositions || !sliceMaxValues || loading) return;

    const nv = nvRef.current;
    if (!nv.volumes || nv.volumes.length === 0) return;

    try {
      // 获取体素维度
      const vol = nv.volumes[0];
      const dims = vol.dims;
      if (!dims || dims.length < 4) return;

      // NIfTI dims: [ndim, x, y, z, ...]
      // 将切片索引转换为归一化的分数坐标 (0-1)
      const fracX = sliceMaxValues.sagittal > 0 ? slicePositions.sagittal / sliceMaxValues.sagittal : 0.5;
      const fracY = sliceMaxValues.coronal > 0 ? slicePositions.coronal / sliceMaxValues.coronal : 0.5;
      const fracZ = sliceMaxValues.axial > 0 ? slicePositions.axial / sliceMaxValues.axial : 0.5;

      // 设置十字光标位置（使用分数坐标）
      nv.scene.crosshairPos = [fracX, fracY, fracZ];
      nv.drawScene();
    } catch (err) {
      console.warn('同步切片位置失败:', err);
    }
  }, [slicePositions, sliceMaxValues, loading]);

  // 切换视图模式
  const toggleViewMode = useCallback(() => {
    if (!nvRef.current) return;
    const nv = nvRef.current;
    if (viewMode === 'render') {
      nv.setSliceType(nv.sliceTypeMultiplanar);
      setViewMode('multiplanar');
    } else {
      nv.setSliceType(nv.sliceTypeRender);
      setViewMode('render');
    }
  }, [viewMode]);

  // 切换分割显示
  const toggleSegmentation = useCallback(() => {
    if (!nvRef.current || nvRef.current.volumes.length < 2) return;
    const newShow = !showSegmentation;
    setShowSegmentation(newShow);
    nvRef.current.setOpacity(1, newShow ? segOpacity : 0);
  }, [showSegmentation, segOpacity]);

  // 更新分割透明度
  const handleOpacityChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newOpacity = parseFloat(e.target.value);
    setSegOpacity(newOpacity);
    if (nvRef.current && nvRef.current.volumes.length > 1 && showSegmentation) {
      nvRef.current.setOpacity(1, newOpacity);
    }
  }, [showSegmentation]);

  // 刷新
  const handleRefresh = useCallback(() => {
    // 清理当前实例
    if (nvRef.current) {
      try { nvRef.current.closeAllVolumes?.(); } catch { /* ignore */ }
      nvRef.current = null;
    }
    currentSessionRef.current = null;
    setError(null);
    // 增加刷新计数器触发 useEffect 重新执行
    setRefreshKey(prev => prev + 1);
  }, []);

  const hasSegmentation = nvRef.current?.volumes?.length > 1;

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* 标题栏 */}
      <div className="flex items-center justify-between px-2 py-1 bg-gray-800 border-b border-gray-700 shrink-0">
        <span className="text-xs font-medium text-gray-300">3D</span>
        <div className="flex items-center gap-1">
          {/* 十字光标同步开关 */}
          <button
            onClick={() => {
              const newShow = !showCrosshair;
              setShowCrosshair(newShow);
              if (nvRef.current) {
                nvRef.current.opts.show3Dcrosshair = newShow;
                nvRef.current.drawScene();
              }
            }}
            className={`p-1 rounded ${showCrosshair ? 'bg-yellow-600' : 'bg-gray-700'} hover:opacity-80`}
            title={showCrosshair ? '隐藏同步十字光标' : '显示同步十字光标'}
          >
            <Crosshair className="w-3 h-3 text-white" />
          </button>
          {/* 分割显示开关 */}
          {hasSegmentation && (
            <button
              onClick={toggleSegmentation}
              className={`p-1 rounded ${showSegmentation ? 'bg-green-600' : 'bg-gray-700'} hover:opacity-80`}
              title={showSegmentation ? '隐藏分割' : '显示分割'}
            >
              {showSegmentation ? <Eye className="w-3 h-3 text-white" /> : <EyeOff className="w-3 h-3 text-white" />}
            </button>
          )}
          <button
            onClick={toggleViewMode}
            className={`p-1 rounded ${viewMode === 'render' ? 'bg-blue-600' : 'bg-gray-700'} hover:opacity-80`}
            title={viewMode === 'render' ? '多平面' : '3D渲染'}
          >
            {viewMode === 'render' ? <Box className="w-3 h-3 text-white" /> : <Layers className="w-3 h-3 text-white" />}
          </button>
          <button onClick={handleRefresh} className="p-1 hover:bg-gray-700 rounded" title="刷新">
            <RefreshCw className="w-3 h-3 text-gray-400" />
          </button>
        </div>
      </div>

      {/* 分割透明度控制 */}
      {hasSegmentation && showSegmentation && !loading && (
        <div className="px-2 py-1 bg-gray-800/50 border-b border-gray-700 flex items-center gap-2 shrink-0">
          <span className="text-[10px] text-gray-400">分割</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={segOpacity}
            onChange={handleOpacityChange}
            className="flex-1 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
          />
          <span className="text-[10px] text-gray-400 w-8">{Math.round(segOpacity * 100)}%</span>
        </div>
      )}

      {/* 3D 画布 */}
      <div className="flex-1 relative bg-black min-h-0">
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10">
            <Loader2 className="w-6 h-6 animate-spin text-blue-400 mb-2" />
            <span className="text-xs text-gray-400 mb-1">{progress.stage}</span>
            {progress.percent > 0 && (
              <div className="w-32 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-200"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
            )}
            {progress.percent > 0 && (
              <span className="text-[10px] text-gray-500 mt-1">{progress.percent}%</span>
            )}
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-4 z-10">
            <AlertCircle className="w-6 h-6 text-red-400 mb-2" />
            <span className="text-xs text-red-400">{error}</span>
          </div>
        )}
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
}

