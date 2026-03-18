'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Loader2 } from 'lucide-react';

interface Task {
  session_id: string;
  filename: string;
  file_type?: string;
  status: string;
  task_type?: string;
  shape?: number[];
  organs?: string[];
  device_used?: string;
  error_message?: string;
  created_at?: string;
  completed_at?: string;
}

interface MemoryDebugTopItem {
  session_id: string;
  status?: string;
  task?: string;
  total_bytes: number;
  total_human: string;
  last_access_iso?: string | null;
}

interface MemoryDebugResponse {
  status: string;
  process: {
    rss_bytes?: number | null;
    rss_human: string;
    rss_source?: string;
  };
  sessions: {
    total: number;
    cached_with_arrays: number;
    runtime_arrays_total_bytes: number;
    runtime_arrays_total_human: string;
  };
  top: MemoryDebugTopItem[];
}

interface MemoryPruneResponse {
  status: string;
  prune?: {
    released?: {
      count?: number;
      bytes_human?: string;
    };
  };
  process_rss?: {
    released_human?: string;
  };
}

// 活动状态类型
type ActivityStatus = 'idle' | 'generating_preview' | 'exporting';

// 会话活动状态
interface SessionActivity {
  sessionId: string;
  activity: ActivityStatus;
}

interface TaskListProps {
  onTaskSelect?: (sessionId: string) => void;
  refreshTrigger?: number;
  sessionActivity?: SessionActivity | null;  // 当前会话的活动状态
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:28000';

// 状态颜色映射
const statusColors: Record<string, string> = {
  completed: 'bg-green-100 text-green-800',
  processing: 'bg-blue-100 text-blue-800',
  uploaded: 'bg-yellow-100 text-yellow-800',
  error: 'bg-red-100 text-red-800',
  unknown: 'bg-gray-100 text-gray-800',
  generating_preview: 'bg-purple-100 text-purple-800',
  exporting: 'bg-orange-100 text-orange-800',
};

// 状态中文映射
const statusLabels: Record<string, string> = {
  completed: '已完成',
  processing: '处理中',
  uploaded: '已上传',
  error: '错误',
  unknown: '未知',
  generating_preview: '生成预览中',
  exporting: '导出中',
};

// 获取任务的显示状态（考虑活动状态 - 优先使用前端传递的，其次使用服务端轮询的）
function getDisplayStatus(
  task: Task,
  sessionActivity?: SessionActivity | null,
  serverActivities?: Record<string, { activity_status: ActivityStatus }>
): string {
  // 优先使用前端传递的活动状态
  if (sessionActivity && sessionActivity.sessionId === task.session_id && sessionActivity.activity !== 'idle') {
    return sessionActivity.activity;
  }
  // 其次使用服务端轮询的活动状态
  if (serverActivities && serverActivities[task.session_id]) {
    const serverActivity = serverActivities[task.session_id].activity_status;
    if (serverActivity !== 'idle') {
      return serverActivity;
    }
  }
  return task.status;
}

export default function TaskList({ onTaskSelect, refreshTrigger, sessionActivity }: TaskListProps) {
  const router = useRouter();
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [initialLoading, setInitialLoading] = useState(true);  // 首次加载标记
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [serverActivities, setServerActivities] = useState<Record<string, { activity_status: ActivityStatus }>>({});
  const [memoryDebug, setMemoryDebug] = useState<MemoryDebugResponse | null>(null);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryPruning, setMemoryPruning] = useState(false);
  const [memoryPanelOpen, setMemoryPanelOpen] = useState(false);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [memoryActionMessage, setMemoryActionMessage] = useState<string | null>(null);
  const pageSize = 10;

  // 加载任务列表
  const loadTasks = useCallback(async () => {
    // 仅首次加载时显示loading状态（骨架屏）
    if (initialLoading) {
      setLoading(true);
    }
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tasks?page=${page}&page_size=${pageSize}`);
      if (!res.ok) throw new Error('加载失败');
      const data = await res.json();
      setTasks(data.tasks || []);
      setTotalPages(data.total_pages || 1);
      setTotal(data.total || 0);
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败');
    } finally {
      setLoading(false);
      setInitialLoading(false);
    }
  }, [page, initialLoading]);

  // 加载服务端活动状态
  const loadActivities = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions/activities`);
      if (res.ok) {
        const data = await res.json();
        setServerActivities(data.activities || {});
      }
    } catch (e) {
      console.error('加载活动状态失败:', e);
    }
  }, []);

  // 加载内存诊断数据
  const loadMemoryDebug = useCallback(async (topN: number = 10) => {
    setMemoryLoading(true);
    setMemoryError(null);
    try {
      const res = await fetch(`${API_BASE}/api/debug/memory?top_n=${topN}&include_idle=false`);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || '内存诊断加载失败');
      setMemoryDebug(data);
      setMemoryPanelOpen(true);
    } catch (e) {
      setMemoryError(e instanceof Error ? e.message : '内存诊断加载失败');
      setMemoryPanelOpen(true);
    } finally {
      setMemoryLoading(false);
    }
  }, []);

  // 一键清理运行时内存缓存
  const handlePruneMemory = useCallback(async () => {
    setMemoryPruning(true);
    setMemoryError(null);
    setMemoryActionMessage(null);

    try {
      const res = await fetch(`${API_BASE}/api/debug/memory/prune?keep_latest_sessions=1`, {
        method: 'POST',
      });
      const data: MemoryPruneResponse = await res.json();
      if (!res.ok) throw new Error((data as { detail?: string })?.detail || '内存清理失败');

      const releasedCount = data.prune?.released?.count ?? 0;
      const releasedHuman = data.prune?.released?.bytes_human || 'unknown';
      const rssReleasedHuman = data.process_rss?.released_human;

      setMemoryActionMessage(
        rssReleasedHuman && rssReleasedHuman !== 'unknown'
          ? `清理完成：释放 ${releasedCount} 个会话缓存（${releasedHuman}），进程 RSS 下降 ${rssReleasedHuman}`
          : `清理完成：释放 ${releasedCount} 个会话缓存（${releasedHuman}）`
      );

      await loadMemoryDebug(10);
    } catch (e) {
      setMemoryError(e instanceof Error ? e.message : '内存清理失败');
      setMemoryPanelOpen(true);
    } finally {
      setMemoryPruning(false);
    }
  }, [loadMemoryDebug]);

  useEffect(() => {
    loadTasks();
  }, [loadTasks, refreshTrigger]);

  // 定期轮询活动状态（每 2 秒）
  useEffect(() => {
    loadActivities();
    const interval = setInterval(loadActivities, 2000);
    return () => clearInterval(interval);
  }, [loadActivities]);

  // 选择/取消选择
  const toggleSelect = (id: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedIds(newSet);
  };

  // 全选/取消全选
  const toggleSelectAll = () => {
    if (selectedIds.size === tasks.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(tasks.map(t => t.session_id)));
    }
  };

  // 查看任务
  const handleView = (sessionId: string) => {
    if (onTaskSelect) {
      onTaskSelect(sessionId);
    } else {
      router.push(`/viewer/${sessionId}`);
    }
  };

  // 删除单个任务
  const handleDelete = (sessionId: string) => {
    setDeleteTarget(sessionId);
    setShowDeleteConfirm(true);
  };

  // 批量删除
  const handleBatchDelete = () => {
    if (selectedIds.size === 0) return;
    setDeleteTarget(null);
    setShowDeleteConfirm(true);
  };

  // 确认删除
  const confirmDelete = async () => {
    setDeleting(true);
    try {
      if (deleteTarget) {
        // 单个删除
        const res = await fetch(`${API_BASE}/api/tasks/${deleteTarget}?confirm=true`, {
          method: 'DELETE',
        });
        if (!res.ok) throw new Error('删除失败');
      } else {
        // 批量删除
        const res = await fetch(`${API_BASE}/api/tasks/batch-delete`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_ids: Array.from(selectedIds),
            confirm: true,
          }),
        });
        if (!res.ok) throw new Error('批量删除失败');
        setSelectedIds(new Set());
      }
      await loadTasks();
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除失败');
    } finally {
      setDeleting(false);
      setShowDeleteConfirm(false);
      setDeleteTarget(null);
    }
  };

  // 格式化时间
  const formatTime = (timeStr?: string) => {
    if (!timeStr) return '-';
    const date = new Date(timeStr);
    return date.toLocaleString('zh-CN');
  };

  // 仅在首次加载且无数据时显示骨架屏
  if (initialLoading && loading && tasks.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse flex space-x-4">
          <div className="flex-1 space-y-4 py-1">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="space-y-2">
              <div className="h-4 bg-gray-200 rounded"></div>
              <div className="h-4 bg-gray-200 rounded w-5/6"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow w-full">
      {/* 头部 */}
      <div className="px-4 sm:px-6 py-3 sm:py-4 border-b border-gray-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="flex items-center gap-3 sm:gap-4">
          <h2 className="text-base sm:text-lg font-semibold text-gray-900">历史任务</h2>
          <span className="text-xs sm:text-sm text-gray-500">共 {total} 个</span>
        </div>
        <div className="flex items-center gap-2">
          {selectedIds.size > 0 && (
            <button
              onClick={handleBatchDelete}
              className="px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
            >
              删除 ({selectedIds.size})
            </button>
          )}
          <button
            onClick={() => void loadMemoryDebug()}
            disabled={memoryLoading || memoryPruning}
            className="px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm bg-indigo-100 text-indigo-700 rounded hover:bg-indigo-200 transition-colors disabled:opacity-50"
          >
            {memoryLoading ? (
              <span className="inline-flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" />
                诊断中
              </span>
            ) : (
              '内存诊断'
            )}
          </button>
          <button
            onClick={() => void handlePruneMemory()}
            disabled={memoryPruning || memoryLoading}
            className="px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm bg-amber-100 text-amber-800 rounded hover:bg-amber-200 transition-colors disabled:opacity-50"
          >
            {memoryPruning ? (
              <span className="inline-flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" />
                清理中
              </span>
            ) : (
              '一键清理缓存'
            )}
          </button>
          <button
            onClick={loadTasks}
            className="px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            刷新
          </button>
        </div>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="px-6 py-3 bg-red-50 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* 内存诊断面板 */}
      {memoryPanelOpen && (
        <div className="px-4 sm:px-6 py-3 border-b border-indigo-100 bg-indigo-50/70">
          <div className="flex items-start sm:items-center justify-between gap-3 mb-2">
            <div>
              <h3 className="text-sm font-semibold text-indigo-900">内存诊断</h3>
              <p className="text-xs text-indigo-700">用于定位会话缓存导致的内存持续增长问题</p>
            </div>
            <button
              onClick={() => setMemoryPanelOpen(false)}
              className="text-xs text-indigo-700 hover:text-indigo-900"
            >
              收起
            </button>
          </div>

          {memoryError && (
            <div className="mb-2 px-3 py-2 text-xs rounded bg-red-50 text-red-700 border border-red-200">
              {memoryError}
            </div>
          )}

          {memoryActionMessage && (
            <div className="mb-2 px-3 py-2 text-xs rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
              {memoryActionMessage}
            </div>
          )}

          {memoryLoading && (
            <div className="text-xs text-indigo-700 inline-flex items-center gap-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              正在读取内存指标...
            </div>
          )}

          {!memoryLoading && memoryDebug && (
            <div className="space-y-2">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                <div className="px-3 py-2 bg-white rounded border border-indigo-100">
                  <div className="text-gray-500">进程 RSS</div>
                  <div className="font-semibold text-gray-900">{memoryDebug.process.rss_human}</div>
                </div>
                <div className="px-3 py-2 bg-white rounded border border-indigo-100">
                  <div className="text-gray-500">运行时数组总占用</div>
                  <div className="font-semibold text-gray-900">{memoryDebug.sessions.runtime_arrays_total_human}</div>
                </div>
                <div className="px-3 py-2 bg-white rounded border border-indigo-100">
                  <div className="text-gray-500">缓存会话</div>
                  <div className="font-semibold text-gray-900">
                    {memoryDebug.sessions.cached_with_arrays} / {memoryDebug.sessions.total}
                  </div>
                </div>
              </div>

              <div className="text-xs text-gray-600">
                Top 会话缓存占用（仅展示有数组缓存的会话）
              </div>

              {memoryDebug.top.length === 0 ? (
                <div className="text-xs text-gray-500">当前没有会话数组缓存。</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs border border-indigo-100 rounded bg-white">
                    <thead className="bg-indigo-100/60 text-gray-700">
                      <tr>
                        <th className="px-2 py-1 text-left">Session</th>
                        <th className="px-2 py-1 text-left">状态</th>
                        <th className="px-2 py-1 text-left">占用</th>
                        <th className="px-2 py-1 text-left">最近访问</th>
                      </tr>
                    </thead>
                    <tbody>
                      {memoryDebug.top.map((item) => (
                        <tr key={item.session_id} className="border-t border-indigo-50">
                          <td className="px-2 py-1 font-mono text-[11px] text-gray-700">{item.session_id.slice(0, 12)}...</td>
                          <td className="px-2 py-1 text-gray-600">{item.status || '-'}</td>
                          <td className="px-2 py-1 text-gray-900 font-medium">{item.total_human}</td>
                          <td className="px-2 py-1 text-gray-600">{formatTime(item.last_access_iso || undefined)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 任务列表 */}
      {tasks.length === 0 ? (
        <div className="px-4 sm:px-6 py-8 sm:py-12 text-center text-gray-500">
          暂无历史任务
        </div>
      ) : (
        <>
          {/* 桌面端表格视图 */}
          <div className="hidden lg:block overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left w-10">
                    <input
                      type="checkbox"
                      checked={selectedIds.size === tasks.length && tasks.length > 0}
                      onChange={toggleSelectAll}
                      className="rounded border-gray-300"
                    />
                  </th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">文件名</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase w-36">状态</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase w-20">类型</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase w-44">创建时间</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase w-20">操作</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {tasks.map((task) => (
                  <tr key={task.session_id} className="hover:bg-gray-50">
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        checked={selectedIds.has(task.session_id)}
                        onChange={() => toggleSelect(task.session_id)}
                        className="rounded border-gray-300"
                      />
                    </td>
                    <td className="px-3 py-2">
                      <div className="text-sm font-medium text-gray-900 truncate max-w-md" title={task.filename}>
                        {task.filename}
                      </div>
                      <div className="text-xs text-gray-500">{task.session_id.slice(0, 8)}...</div>
                    </td>
                    <td className="px-3 py-2">
                      {(() => {
                        const displayStatus = getDisplayStatus(task, sessionActivity, serverActivities);
                        const isActive = displayStatus === 'generating_preview' || displayStatus === 'exporting';
                        return (
                          <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold rounded-full ${statusColors[displayStatus] || statusColors.unknown}`}>
                            {isActive && <Loader2 className="w-3 h-3 animate-spin" />}
                            {statusLabels[displayStatus] || displayStatus}
                          </span>
                        );
                      })()}
                    </td>
                    <td className="px-3 py-2 text-xs text-gray-500">
                      {task.file_type || '-'}
                    </td>
                    <td className="px-3 py-2 text-xs text-gray-500">
                      {formatTime(task.created_at)}
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleView(task.session_id)}
                          disabled={task.status !== 'completed'}
                          className="text-blue-600 hover:text-blue-800 text-xs disabled:text-gray-400 disabled:cursor-not-allowed"
                        >
                          查看
                        </button>
                        <button
                          onClick={() => handleDelete(task.session_id)}
                          className="text-red-600 hover:text-red-800 text-xs"
                        >
                          删除
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* 移动端/平板卡片视图 */}
          <div className="lg:hidden divide-y divide-gray-200">
            {/* 全选控制 */}
            <div className="px-4 py-2 bg-gray-50 flex items-center gap-2">
              <input
                type="checkbox"
                checked={selectedIds.size === tasks.length && tasks.length > 0}
                onChange={toggleSelectAll}
                className="rounded border-gray-300"
              />
              <span className="text-xs text-gray-500">全选</span>
            </div>
            {tasks.map((task) => (
              <div key={task.session_id} className="px-4 py-3 hover:bg-gray-50">
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(task.session_id)}
                    onChange={() => toggleSelect(task.session_id)}
                    className="rounded border-gray-300 mt-1"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <div className="text-sm font-medium text-gray-900 truncate" title={task.filename}>
                        {task.filename}
                      </div>
                      {(() => {
                        const displayStatus = getDisplayStatus(task, sessionActivity, serverActivities);
                        const isActive = displayStatus === 'generating_preview' || displayStatus === 'exporting';
                        return (
                          <span className={`shrink-0 inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold rounded-full ${statusColors[displayStatus] || statusColors.unknown}`}>
                            {isActive && <Loader2 className="w-3 h-3 animate-spin" />}
                            {statusLabels[displayStatus] || displayStatus}
                          </span>
                        );
                      })()}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-gray-500 mb-2">
                      <span>{task.file_type || '-'}</span>
                      <span>{formatTime(task.created_at)}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => handleView(task.session_id)}
                        disabled={task.status !== 'completed'}
                        className="text-blue-600 hover:text-blue-800 text-xs disabled:text-gray-400 disabled:cursor-not-allowed"
                      >
                        查看
                      </button>
                      <button
                        onClick={() => handleDelete(task.session_id)}
                        className="text-red-600 hover:text-red-800 text-xs"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* 分页 */}
      {totalPages > 1 && (
        <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
          <div className="text-sm text-gray-500">
            第 {page} / {totalPages} 页
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 text-sm border rounded disabled:opacity-50"
            >
              上一页
            </button>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-3 py-1 text-sm border rounded disabled:opacity-50"
            >
              下一页
            </button>
          </div>
        </div>
      )}

      {/* 删除确认对话框 */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              ⚠️ 确认删除
            </h3>
            <p className="text-gray-600 mb-4">
              {deleteTarget
                ? '确定要删除这个任务吗？'
                : `确定要删除选中的 ${selectedIds.size} 个任务吗？`}
            </p>
            <p className="text-red-600 text-sm mb-6">
              此操作将永久删除所有相关数据（包括文件、分割结果等），无法恢复！
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowDeleteConfirm(false);
                  setDeleteTarget(null);
                }}
                disabled={deleting}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded hover:bg-gray-200"
              >
                取消
              </button>
              <button
                onClick={confirmDelete}
                disabled={deleting}
                className="px-4 py-2 text-white bg-red-600 rounded hover:bg-red-700 disabled:opacity-50"
              >
                {deleting ? '删除中...' : '确认删除'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
