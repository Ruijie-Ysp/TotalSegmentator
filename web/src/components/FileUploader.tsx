'use client';

import React, { useCallback, useState } from 'react';
import { Upload, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

interface FileUploaderProps {
  onUpload: (file: File) => Promise<void>;
  isProcessing: boolean;
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error' | 'restoring';
  statusMessage?: string;
}

export default function FileUploader({
  onUpload,
  isProcessing,
  status,
  statusMessage,
}: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) {
        await onUpload(file);
      }
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        await onUpload(file);
      }
    },
    [onUpload]
  );

  const statusIcons = {
    idle: <Upload className="w-12 h-12 text-gray-400" />,
    uploading: <Loader2 className="w-12 h-12 text-blue-400 animate-spin" />,
    processing: <Loader2 className="w-12 h-12 text-orange-400 animate-spin" />,
    completed: <CheckCircle className="w-12 h-12 text-green-400" />,
    error: <XCircle className="w-12 h-12 text-red-400" />,
    restoring: <Loader2 className="w-12 h-12 text-purple-400 animate-spin" />,
  };

  const statusTexts = {
    idle: '拖放文件到这里，或点击选择',
    uploading: '正在上传...',
    processing: '正在分割处理...',
    completed: '处理完成！',
    error: statusMessage || '发生错误',
    restoring: '正在恢复会话...',
  };

  return (
    <div
      className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
        isDragging
          ? 'border-blue-500 bg-blue-500/10'
          : status === 'error'
          ? 'border-red-500 bg-red-500/5'
          : status === 'completed'
          ? 'border-green-500 bg-green-500/5'
          : 'border-gray-600 bg-gray-800/50 hover:border-gray-500'
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept=".nii,.nii.gz,.zip"
        onChange={handleFileSelect}
        disabled={isProcessing}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
      />

      <div className="flex flex-col items-center gap-4">
        {statusIcons[status]}
        <div>
          <p className="text-lg font-medium text-gray-200">{statusTexts[status]}</p>
          <p className="text-sm text-gray-400 mt-1">
            支持格式: NIfTI (.nii, .nii.gz), DICOM序列 (ZIP压缩包)
          </p>
        </div>

        {status === 'idle' && (
          <>
            <button
              disabled={isProcessing}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium transition-colors disabled:opacity-50"
            >
              选择文件
            </button>

            {/* 重要提示 */}
            <div className="mt-2 p-3 bg-amber-900/30 border border-amber-600/50 rounded-lg max-w-md">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div className="text-left">
                  <p className="text-sm font-medium text-amber-300">重要提示</p>
                  <p className="text-xs text-amber-200/80 mt-1">
                    TotalSegmentator 需要完整的 <strong>3D CT/MR 序列影像</strong>，
                    不支持单张 DICOM 或 2D 图片。请上传包含多张切片的 DICOM 序列（ZIP压缩包）或 NIfTI 格式的 3D 体积数据。
                  </p>
                </div>
              </div>
            </div>
          </>
        )}

        {statusMessage && status !== 'error' && (
          <p className="text-sm text-gray-400">{statusMessage}</p>
        )}
      </div>

      {/* 进度指示 */}
      {(status === 'uploading' || status === 'processing') && (
        <div className="mt-4">
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-500 ${
                status === 'uploading' ? 'bg-blue-500 w-1/3' : 'bg-orange-500 w-2/3'
              }`}
            />
          </div>
        </div>
      )}
    </div>
  );
}

