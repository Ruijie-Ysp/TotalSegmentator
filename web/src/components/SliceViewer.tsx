'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getSliceUrl, getOrganLabels, OrganLabel, OverlayMode } from '@/lib/api';
import { Loader2, ZoomIn, ZoomOut, RefreshCw, Tag, Maximize2 } from 'lucide-react';
import { getOrganDisplayName } from '@/lib/organs';

interface SliceViewerProps {
  sessionId: string;
  axis: 'axial' | 'sagittal' | 'coronal';
  maxSlice: number;
  defaultSlice: number;
  windowCenter: number;
  windowWidth: number;
  selectedOrgans: string[];
  overlayOpacity: number;
  overlayMode: OverlayMode;
  contourThickness: number;
  showLabels?: boolean;  // 是否显示器官标签
  onSliceChange?: (slice: number) => void;
}

export default function SliceViewer({
  sessionId,
  axis,
  maxSlice,
  defaultSlice,
  windowCenter,
  windowWidth,
  selectedOrgans,
  overlayOpacity,
  overlayMode,
  contourThickness,
  showLabels = true,
  onSliceChange,
}: SliceViewerProps) {
  const [sliceIdx, setSliceIdx] = useState(defaultSlice);
  const [loading, setLoading] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [imageUrl, setImageUrl] = useState('');
  const [labels, setLabels] = useState<OrganLabel[]>([]);
  const [showLabelToggle, setShowLabelToggle] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const axisLabels = {
    axial: '轴向 (Axial)',
    sagittal: '矢状 (Sagittal)',
    coronal: '冠状 (Coronal)',
  };

  // 生成图片 URL（添加 refreshKey 和时间戳强制刷新）
  useEffect(() => {
    if (sessionId && maxSlice > 0) {
      const url = getSliceUrl(
        sessionId,
        axis,
        sliceIdx,
        windowCenter,
        windowWidth,
        selectedOrgans,
        overlayOpacity,
        overlayMode,
        contourThickness
      );
      // 添加时间戳避免缓存
      const urlWithCache = `${url}&_t=${Date.now()}`;
      setImageUrl(urlWithCache);
      setLoading(true);
    }
  }, [sessionId, axis, sliceIdx, windowCenter, windowWidth, selectedOrgans, overlayOpacity, overlayMode, contourThickness, maxSlice, refreshKey]);

  // 刷新重载数据
  const handleRefresh = useCallback(() => {
    setRefreshKey(k => k + 1);
  }, []);

  // 获取器官标签位置
  useEffect(() => {
    if (sessionId && selectedOrgans.length > 0 && showLabels && showLabelToggle) {
      getOrganLabels(sessionId, axis, sliceIdx, selectedOrgans)
        .then((res) => setLabels(res.labels))
        .catch(() => setLabels([]));
    } else {
      setLabels([]);
    }
  }, [sessionId, axis, sliceIdx, selectedOrgans, showLabels, showLabelToggle]);

  // 键盘导航
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') {
      setSliceIdx((prev) => Math.min(prev + 1, maxSlice - 1));
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') {
      setSliceIdx((prev) => Math.max(prev - 1, 0));
    }
  }, [maxSlice]);

  // 鼠标滚轮 - 按住 Ctrl 缩放，否则切换切片
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      // 缩放
      const delta = e.deltaY > 0 ? -0.25 : 0.25;
      setZoom(z => Math.max(0.5, Math.min(5, z + delta)));
    } else {
      // 切换切片
      const delta = e.deltaY > 0 ? -1 : 1;
      setSliceIdx((prev) => Math.max(0, Math.min(prev + delta, maxSlice - 1)));
    }
  }, [maxSlice]);

  // 拖拽移动
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  }, [position]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // 双击重置视图
  const handleDoubleClick = useCallback(() => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  }, []);

  // 重置视图
  const resetView = useCallback(() => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  }, []);

  // 全局鼠标释放监听
  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  // 使用 ref 存储回调，避免依赖变化导致无限循环
  const onSliceChangeRef = useRef(onSliceChange);
  onSliceChangeRef.current = onSliceChange;

  useEffect(() => {
    onSliceChangeRef.current?.(sliceIdx);
  }, [sliceIdx]);

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* 标题栏 - 紧凑设计 */}
      <div className="flex items-center justify-between px-2 py-1.5 bg-gray-800 border-b border-gray-700 shrink-0">
        <span className="text-xs font-medium text-gray-200">{axisLabels[axis]}</span>
        <div className="flex items-center gap-1">
          {/* 标签开关 */}
          <button
            onClick={() => setShowLabelToggle(!showLabelToggle)}
            className={`p-0.5 rounded ${showLabelToggle ? 'bg-blue-600 hover:bg-blue-700' : 'hover:bg-gray-700'}`}
            title={showLabelToggle ? '隐藏标签' : '显示标签'}
          >
            <Tag className="w-3.5 h-3.5 text-gray-200" />
          </button>
          <div className="w-px h-3 bg-gray-600 mx-0.5" />
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            className="p-0.5 hover:bg-gray-700 rounded"
            title="缩小"
          >
            <ZoomOut className="w-3.5 h-3.5 text-gray-400" />
          </button>
          <span className="text-xs text-gray-400 w-8 text-center">{Math.round(zoom * 100)}%</span>
          <button
            onClick={() => setZoom((z) => Math.min(5, z + 0.25))}
            className="p-0.5 hover:bg-gray-700 rounded"
            title="放大"
          >
            <ZoomIn className="w-3.5 h-3.5 text-gray-400" />
          </button>
          <button
            onClick={() => { resetView(); handleRefresh(); }}
            className="p-0.5 hover:bg-gray-700 rounded"
            title="重置视图"
          >
            <Maximize2 className="w-3.5 h-3.5 text-gray-400" />
          </button>
          <button
            onClick={handleRefresh}
            className="p-0.5 hover:bg-gray-700 rounded"
            title="刷新"
          >
            <RefreshCw className="w-3.5 h-3.5 text-gray-400" />
          </button>
        </div>
      </div>

      {/* 图像区域 - 支持拖拽移动 */}
      <div
        ref={containerRef}
        className={`relative flex-1 min-h-0 bg-black overflow-hidden ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        tabIndex={0}
        onKeyDown={(e) => handleKeyDown(e.nativeEvent)}
      >
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
            <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
          </div>
        )}
        {imageUrl && (
          <div
            className="absolute inset-0 flex items-center justify-center"
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`,
              transformOrigin: 'center',
              transition: isDragging ? 'none' : 'transform 0.1s ease-out',
            }}
          >
            <img
              ref={imageRef}
              src={imageUrl}
              alt={`${axis} slice ${sliceIdx}`}
              className="max-w-full max-h-full object-contain"
              onLoad={() => setLoading(false)}
              onError={() => setLoading(false)}
              draggable={false}
            />
            {/* 器官标签覆盖层 */}
            {showLabelToggle && labels.length > 0 && imageRef.current && (
              <div className="absolute inset-0 pointer-events-none">
                {labels.map((label) => (
                  <div
                    key={label.organ}
                    className="absolute transform -translate-x-1/2 -translate-y-1/2"
                    style={{
                      left: `${label.x * 100}%`,
                      top: `${label.y * 100}%`,
                    }}
                  >
                    <div
                      className="px-1.5 py-0.5 rounded text-[10px] font-medium whitespace-nowrap shadow-lg"
                      style={{
                        backgroundColor: `rgba(${label.color[0]}, ${label.color[1]}, ${label.color[2]}, 0.85)`,
                        color: label.color[0] + label.color[1] + label.color[2] > 400 ? '#000' : '#fff',
                        border: `1px solid rgba(${label.color[0]}, ${label.color[1]}, ${label.color[2]}, 1)`,
                      }}
                    >
                      {getOrganDisplayName(label.organ)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 切片滑条 - 紧凑设计 */}
      <div className="px-2 py-1.5 bg-gray-800 border-t border-gray-700 shrink-0">
        <div className="flex items-center gap-2">
          <input
            type="range"
            min={0}
            max={maxSlice - 1}
            value={sliceIdx}
            onChange={(e) => setSliceIdx(parseInt(e.target.value))}
            className="flex-1 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <span className="text-xs text-gray-300 w-16 text-right tabular-nums">
            {sliceIdx + 1}/{maxSlice}
          </span>
        </div>
      </div>
    </div>
  );
}

