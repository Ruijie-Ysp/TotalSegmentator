'use client';

import React from 'react';
import { WindowPreset, OverlayMode } from '@/lib/api';
import { Settings, Download, FileJson, Layers, PenTool, Sparkles } from 'lucide-react';
import { getTaskDisplayName } from '@/lib/organs';

interface ControlPanelProps {
  windowCenter: number;
  windowWidth: number;
  overlayOpacity: number;
  overlayMode: OverlayMode;
  contourThickness: number;
  windowPresets: Record<string, WindowPreset>;
  onWindowChange: (center: number, width: number) => void;
  onOpacityChange: (opacity: number) => void;
  onOverlayModeChange: (mode: OverlayMode) => void;
  onContourThicknessChange: (thickness: number) => void;
  onExportNifti: () => void;
  onExportLabelStudio: () => void;
  onExportTraining?: () => void;
  task: string;
  onTaskChange: (task: string) => void;
  availableTasks: string[];
  isProcessing: boolean;
}

const OVERLAY_MODES: { value: OverlayMode; label: string; icon: string }[] = [
  { value: 'fill', label: '填充', icon: '■' },
  { value: 'contour', label: '轮廓', icon: '□' },
  { value: 'fill_contour', label: '填充+轮廓', icon: '▣' },
];

export default function ControlPanel({
  windowCenter,
  windowWidth,
  overlayOpacity,
  overlayMode,
  contourThickness,
  windowPresets,
  onWindowChange,
  onOpacityChange,
  onOverlayModeChange,
  onContourThicknessChange,
  onExportNifti,
  onExportLabelStudio,
  onExportTraining,
  task,
  onTaskChange,
  availableTasks,
  isProcessing,
}: ControlPanelProps) {
  return (
    <div className="flex flex-col gap-2 p-2 bg-gray-900 rounded-lg border border-gray-700 h-full overflow-y-auto text-xs">
      {/* 任务选择 */}
      <select
        value={task}
        onChange={(e) => onTaskChange(e.target.value)}
        disabled={isProcessing}
        className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-gray-200 text-xs"
      >
        {availableTasks.map((t) => (
          <option key={t} value={t}>{getTaskDisplayName(t)}</option>
        ))}
      </select>

      {/* 窗宽窗位预设 */}
      <div className="flex flex-wrap gap-1">
        {Object.entries(windowPresets).map(([key, preset]) => (
          <button
            key={key}
            onClick={() => onWindowChange(preset.center, preset.width)}
            className={`px-1.5 py-0.5 text-[10px] rounded border ${
              windowCenter === preset.center && windowWidth === preset.width
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-gray-800 border-gray-600 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* 窗位 */}
      <div>
        <div className="flex justify-between text-gray-400 mb-0.5">
          <span>窗位</span><span>{windowCenter}</span>
        </div>
        <input
          type="range" min={-1000} max={1000} value={windowCenter}
          onChange={(e) => onWindowChange(parseInt(e.target.value), windowWidth)}
          className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
      </div>

      {/* 窗宽 */}
      <div>
        <div className="flex justify-between text-gray-400 mb-0.5">
          <span>窗宽</span><span>{windowWidth}</span>
        </div>
        <input
          type="range" min={1} max={4000} value={windowWidth}
          onChange={(e) => onWindowChange(windowCenter, parseInt(e.target.value))}
          className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
      </div>

      {/* 叠加模式 */}
      <div className="flex gap-0.5">
        {OVERLAY_MODES.map((mode) => (
          <button
            key={mode.value}
            onClick={() => onOverlayModeChange(mode.value)}
            className={`flex-1 py-1 text-[10px] rounded border ${
              overlayMode === mode.value
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-gray-800 border-gray-600 text-gray-400 hover:bg-gray-700'
            }`}
            title={mode.label}
          >
            {mode.icon}
          </button>
        ))}
      </div>

      {/* 透明度 */}
      <div>
        <div className="flex justify-between text-gray-400 mb-0.5">
          <span>透明度</span><span>{Math.round(overlayOpacity * 100)}%</span>
        </div>
        <input
          type="range" min={0} max={100} value={overlayOpacity * 100}
          onChange={(e) => onOpacityChange(parseInt(e.target.value) / 100)}
          className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
      </div>

      {/* 轮廓粗细 */}
      {(overlayMode === 'contour' || overlayMode === 'fill_contour') && (
        <div>
          <div className="flex justify-between text-gray-400 mb-0.5">
            <span>轮廓</span><span>{contourThickness}px</span>
          </div>
          <input
            type="range" min={1} max={5} value={contourThickness}
            onChange={(e) => onContourThicknessChange(parseInt(e.target.value))}
            className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>
      )}

      {/* 导出按钮 */}
      <div className="pt-2 border-t border-gray-700 space-y-1 mt-auto">
        <button
          onClick={onExportNifti}
          className="w-full flex items-center justify-center gap-1 px-2 py-1.5 bg-green-600 hover:bg-green-700 rounded text-white text-xs"
        >
          <Download className="w-3 h-3" /> 医学影像导出
        </button>
        <button
          onClick={onExportLabelStudio}
          className="w-full flex items-center justify-center gap-1 px-2 py-1.5 bg-purple-600 hover:bg-purple-700 rounded text-white text-xs"
        >
          <FileJson className="w-3 h-3" /> 语料标注导出
        </button>
        {onExportTraining && (
          <button
            onClick={onExportTraining}
            className="w-full flex items-center justify-center gap-1 px-2 py-1.5 bg-amber-600 hover:bg-amber-700 rounded text-white text-xs"
          >
            <Sparkles className="w-3 h-3" /> 训练数据导出
          </button>
        )}
      </div>
    </div>
  );
}

