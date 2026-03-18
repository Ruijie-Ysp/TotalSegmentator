'use client';

import React, { useMemo } from 'react';
import { ORGAN_GROUPS, getOrganDisplayName, getOrganGroup } from '@/lib/organs';
import { ChevronDown, ChevronRight, Eye, EyeOff, Star } from 'lucide-react';

// 主要器官列表（常用的重要器官）
const PRIMARY_ORGANS = [
  // 腹部主要器官
  'liver', 'spleen', 'pancreas', 'gallbladder', 'stomach',
  // 泌尿系统
  'kidney_left', 'kidney_right', 'urinary_bladder',
  // 心血管
  'heart', 'aorta',
  // 肺部
  'lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
  'lung_middle_lobe_right', 'lung_lower_lobe_right',
  // 其他重要器官
  'brain', 'esophagus', 'trachea',
];

interface OrganSelectorProps {
  availableOrgans: string[];
  selectedOrgans: string[];
  onSelectionChange: (organs: string[]) => void;
}

export default function OrganSelector({
  availableOrgans,
  selectedOrgans,
  onSelectionChange,
}: OrganSelectorProps) {
  const [expandedGroups, setExpandedGroups] = React.useState<Set<string>>(new Set(['abdominal', 'lung']));

  // 按分组整理器官
  const groupedOrgans = useMemo(() => {
    const groups: Record<string, string[]> = {};
    const ungrouped: string[] = [];

    availableOrgans.forEach((organ) => {
      const groupKey = getOrganGroup(organ);
      if (groupKey) {
        if (!groups[groupKey]) groups[groupKey] = [];
        groups[groupKey].push(organ);
      } else {
        ungrouped.push(organ);
      }
    });

    if (ungrouped.length > 0) {
      groups['other'] = ungrouped;
    }

    return groups;
  }, [availableOrgans]);

  const toggleGroup = (groupKey: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupKey)) {
        next.delete(groupKey);
      } else {
        next.add(groupKey);
      }
      return next;
    });
  };

  const toggleOrgan = (organ: string) => {
    if (selectedOrgans.includes(organ)) {
      onSelectionChange(selectedOrgans.filter((o) => o !== organ));
    } else {
      onSelectionChange([...selectedOrgans, organ]);
    }
  };

  const toggleGroupOrgans = (organs: string[], select: boolean) => {
    if (select) {
      const newSelection = [...new Set([...selectedOrgans, ...organs])];
      onSelectionChange(newSelection);
    } else {
      onSelectionChange(selectedOrgans.filter((o) => !organs.includes(o)));
    }
  };

  const selectAll = () => onSelectionChange([...availableOrgans]);
  const clearAll = () => onSelectionChange([]);

  // 只选择主要器官（在可用器官中筛选）
  const selectPrimary = () => {
    const primaryAvailable = availableOrgans.filter((o) => PRIMARY_ORGANS.includes(o));
    onSelectionChange(primaryAvailable);
  };

  // 检查当前是否为主要器官模式
  const isPrimaryMode = useMemo(() => {
    const primaryAvailable = availableOrgans.filter((o) => PRIMARY_ORGANS.includes(o));
    return primaryAvailable.length > 0 &&
           selectedOrgans.length === primaryAvailable.length &&
           primaryAvailable.every((o) => selectedOrgans.includes(o));
  }, [availableOrgans, selectedOrgans]);

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      {/* 标题栏 - 更紧凑 */}
      <div className="flex items-center justify-between px-2 py-1 bg-gray-800 border-b border-gray-700">
        <span className="text-xs font-medium text-gray-300">器官</span>
        <div className="flex items-center gap-1">
          <button
            onClick={selectPrimary}
            className={`text-[10px] px-1.5 py-0.5 rounded text-white flex items-center gap-0.5 ${
              isPrimaryMode ? 'bg-amber-600 hover:bg-amber-700' : 'bg-amber-600/70 hover:bg-amber-600'
            }`}
            title="只显示主要器官（肝、脾、肾、心、肺等）"
          >
            <Star className="w-2.5 h-2.5" />
            主要
          </button>
          <button onClick={selectAll} className="text-[10px] px-1.5 py-0.5 bg-blue-600 hover:bg-blue-700 rounded text-white">全选</button>
          <button onClick={clearAll} className="text-[10px] px-1.5 py-0.5 bg-gray-600 hover:bg-gray-700 rounded text-white">清空</button>
        </div>
      </div>

      {/* 分组列表 */}
      <div className="flex-1 overflow-y-auto p-1 space-y-0.5">
        {Object.entries(groupedOrgans).map(([groupKey, organs]) => {
          const groupInfo = ORGAN_GROUPS[groupKey] || { name: '其他', color: '#888888' };
          const isExpanded = expandedGroups.has(groupKey);
          const selectedCount = organs.filter((o) => selectedOrgans.includes(o)).length;
          const allSelected = selectedCount === organs.length;
          const someSelected = selectedCount > 0 && selectedCount < organs.length;

          return (
            <div key={groupKey} className="rounded overflow-hidden">
              {/* 分组头 */}
              <div
                className="flex items-center gap-1 px-1.5 py-1 bg-gray-800 hover:bg-gray-750 cursor-pointer"
                onClick={() => toggleGroup(groupKey)}
              >
                {isExpanded ? <ChevronDown className="w-3 h-3 text-gray-400" /> : <ChevronRight className="w-3 h-3 text-gray-400" />}
                <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: groupInfo.color }} />
                <span className="flex-1 text-xs text-gray-200 truncate">{groupInfo.name}</span>
                <span className="text-[10px] text-gray-500">{selectedCount}/{organs.length}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); toggleGroupOrgans(organs, !allSelected); }}
                  className="p-0.5 hover:bg-gray-700 rounded"
                >
                  {allSelected ? <Eye className="w-3 h-3 text-blue-400" /> : someSelected ? <Eye className="w-3 h-3 text-gray-400" /> : <EyeOff className="w-3 h-3 text-gray-500" />}
                </button>
              </div>

              {/* 器官列表 */}
              {isExpanded && (
                <div className="pl-4 py-0.5 space-y-0 bg-gray-850">
                  {organs.map((organ) => {
                    const isSelected = selectedOrgans.includes(organ);
                    return (
                      <label key={organ} className="flex items-center gap-1.5 px-1 py-0.5 hover:bg-gray-800 rounded cursor-pointer">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleOrgan(organ)}
                          className="w-3 h-3 rounded border-gray-600 bg-gray-700 text-blue-500"
                        />
                        <span className={`text-[11px] truncate ${isSelected ? 'text-gray-200' : 'text-gray-400'}`}>
                          {getOrganDisplayName(organ)}
                        </span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 底部统计 */}
      <div className="px-2 py-1 bg-gray-800 border-t border-gray-700 text-center">
        <span className="text-[10px] text-gray-500">{selectedOrgans.length}/{availableOrgans.length} 已选</span>
      </div>
    </div>
  );
}

