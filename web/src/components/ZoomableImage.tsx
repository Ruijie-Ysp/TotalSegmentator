'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

interface ZoomableImageProps {
  src: string;
  alt: string;
  className?: string;
  showControls?: boolean;
  minZoom?: number;
  maxZoom?: number;
  zoomStep?: number;
}

export default function ZoomableImage({
  src,
  alt,
  className = '',
  showControls = true,
  minZoom = 0.5,
  maxZoom = 5,
  zoomStep = 0.25,
}: ZoomableImageProps) {
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // 重置视图
  const resetView = useCallback(() => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  }, []);

  // 缩放
  const handleZoom = useCallback((delta: number) => {
    setZoom(z => Math.max(minZoom, Math.min(maxZoom, z + delta)));
  }, [minZoom, maxZoom]);

  // 鼠标滚轮缩放
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
    handleZoom(delta);
  }, [handleZoom, zoomStep]);

  // 开始拖拽
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return; // 只响应左键
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  }, [position]);

  // 拖拽中
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  }, [isDragging, dragStart]);

  // 结束拖拽
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // 双击重置
  const handleDoubleClick = useCallback(() => {
    resetView();
  }, [resetView]);

  // 全局鼠标释放监听
  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  return (
    <div className={`relative flex flex-col h-full ${className}`}>
      {/* 控制按钮 */}
      {showControls && (
        <div className="absolute top-2 right-2 z-10 flex items-center gap-1 bg-black/60 rounded px-1.5 py-1">
          <button
            onClick={() => handleZoom(-zoomStep)}
            className="p-1 hover:bg-white/20 rounded"
            title="缩小"
          >
            <ZoomOut className="w-4 h-4 text-white" />
          </button>
          <span className="text-xs text-white w-10 text-center">{Math.round(zoom * 100)}%</span>
          <button
            onClick={() => handleZoom(zoomStep)}
            className="p-1 hover:bg-white/20 rounded"
            title="放大"
          >
            <ZoomIn className="w-4 h-4 text-white" />
          </button>
          <div className="w-px h-4 bg-white/30 mx-1" />
          <button
            onClick={resetView}
            className="p-1 hover:bg-white/20 rounded"
            title="重置视图"
          >
            <Maximize2 className="w-4 h-4 text-white" />
          </button>
        </div>
      )}

      {/* 图像容器 */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDoubleClick={handleDoubleClick}
      >
        <div
          className="w-full h-full flex items-center justify-center"
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`,
            transformOrigin: 'center',
            transition: isDragging ? 'none' : 'transform 0.1s ease-out',
          }}
        >
          <img
            src={src}
            alt={alt}
            className="max-w-full max-h-full object-contain select-none"
            draggable={false}
          />
        </div>
      </div>
    </div>
  );
}

