// 器官分组和颜色配置

export const ORGAN_GROUPS: Record<string, { name: string; color: string; organs: string[] }> = {
  abdominal: {
    name: '腹部器官',
    color: '#8B4513',
    organs: ['liver', 'spleen', 'pancreas', 'gallbladder', 'stomach', 'small_bowel', 'duodenum', 'colon'],
  },
  urinary: {
    name: '泌尿系统',
    color: '#DC143C',
    organs: ['kidney_left', 'kidney_right', 'urinary_bladder', 'kidney_cyst_left', 'kidney_cyst_right', 'prostate'],
  },
  lung: {
    name: '肺部',
    color: '#4682B4',
    organs: ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'trachea', 'esophagus'],
  },
  cardiovascular: {
    name: '心血管',
    color: '#FF0000',
    organs: ['heart', 'aorta', 'pulmonary_vein', 'inferior_vena_cava', 'superior_vena_cava', 'portal_vein_and_splenic_vein', 'brachiocephalic_trunk', 'subclavian_artery_right', 'subclavian_artery_left', 'common_carotid_artery_right', 'common_carotid_artery_left', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right', 'atrial_appendage_left', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right'],
  },
  skeleton_upper: {
    name: '上肢骨骼',
    color: '#F5F5F5',
    organs: ['skull', 'humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'clavicula_left', 'clavicula_right', 'sternum', 'costal_cartilages'],
  },
  skeleton_lower: {
    name: '下肢骨骼',
    color: '#D3D3D3',
    organs: ['femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum'],
  },
  spine: {
    name: '脊柱',
    color: '#C0C0C0',
    organs: ['vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1', 'spinal_cord'],
  },
  ribs: {
    name: '肋骨',
    color: '#A9A9A9',
    organs: ['rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12'],
  },
  muscles: {
    name: '肌肉',
    color: '#8B5A2B',
    organs: ['gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right', 'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right'],
  },
  glands: {
    name: '腺体',
    color: '#FF69B4',
    organs: ['adrenal_gland_right', 'adrenal_gland_left', 'thyroid_gland'],
  },
  brain: {
    name: '脑部',
    color: '#FFB6C1',
    organs: ['brain'],
  },
};

// 获取器官所属分组
export function getOrganGroup(organ: string): string | null {
  for (const [groupKey, group] of Object.entries(ORGAN_GROUPS)) {
    if (group.organs.includes(organ)) {
      return groupKey;
    }
  }
  return null;
}

// 获取器官显示名称（中文）
export const ORGAN_NAMES_CN: Record<string, string> = {
  // 腹部器官
  liver: '肝脏',
  spleen: '脾脏',
  pancreas: '胰腺',
  gallbladder: '胆囊',
  stomach: '胃',
  small_bowel: '小肠',
  duodenum: '十二指肠',
  colon: '结肠',

  // 泌尿系统
  kidney_left: '左肾',
  kidney_right: '右肾',
  urinary_bladder: '膀胱',
  kidney_cyst_left: '左肾囊肿',
  kidney_cyst_right: '右肾囊肿',
  prostate: '前列腺',

  // 肺部
  lung_upper_lobe_left: '左肺上叶',
  lung_lower_lobe_left: '左肺下叶',
  lung_upper_lobe_right: '右肺上叶',
  lung_middle_lobe_right: '右肺中叶',
  lung_lower_lobe_right: '右肺下叶',
  lung_left: '左肺',
  lung_right: '右肺',
  trachea: '气管',
  esophagus: '食管',

  // 心血管系统
  heart: '心脏',
  aorta: '主动脉',
  pulmonary_vein: '肺静脉',
  inferior_vena_cava: '下腔静脉',
  superior_vena_cava: '上腔静脉',
  portal_vein_and_splenic_vein: '门静脉和脾静脉',
  brachiocephalic_trunk: '头臂干',
  subclavian_artery_right: '右锁骨下动脉',
  subclavian_artery_left: '左锁骨下动脉',
  common_carotid_artery_right: '右颈总动脉',
  common_carotid_artery_left: '左颈总动脉',
  brachiocephalic_vein_left: '左头臂静脉',
  brachiocephalic_vein_right: '右头臂静脉',
  atrial_appendage_left: '左心耳',
  iliac_artery_left: '左髂动脉',
  iliac_artery_right: '右髂动脉',
  iliac_vena_left: '左髂静脉',
  iliac_vena_right: '右髂静脉',
  pulmonary_artery: '肺动脉',

  // 心脏详细结构
  heart_myocardium: '心肌',
  heart_atrium_left: '左心房',
  heart_ventricle_left: '左心室',
  heart_atrium_right: '右心房',
  heart_ventricle_right: '右心室',

  // 骨骼系统 - 头颈
  skull: '颅骨',

  // 骨骼系统 - 上肢
  humerus_left: '左肱骨',
  humerus_right: '右肱骨',
  scapula_left: '左肩胛骨',
  scapula_right: '右肩胛骨',
  clavicula_left: '左锁骨',
  clavicula_right: '右锁骨',
  sternum: '胸骨',
  costal_cartilages: '肋软骨',

  // 骨骼系统 - 下肢
  femur_left: '左股骨',
  femur_right: '右股骨',
  hip_left: '左髋骨',
  hip_right: '右髋骨',
  sacrum: '骶骨',

  // 脊柱
  vertebrae_S1: '骶椎S1',
  vertebrae_L5: '腰椎L5',
  vertebrae_L4: '腰椎L4',
  vertebrae_L3: '腰椎L3',
  vertebrae_L2: '腰椎L2',
  vertebrae_L1: '腰椎L1',
  vertebrae_T12: '胸椎T12',
  vertebrae_T11: '胸椎T11',
  vertebrae_T10: '胸椎T10',
  vertebrae_T9: '胸椎T9',
  vertebrae_T8: '胸椎T8',
  vertebrae_T7: '胸椎T7',
  vertebrae_T6: '胸椎T6',
  vertebrae_T5: '胸椎T5',
  vertebrae_T4: '胸椎T4',
  vertebrae_T3: '胸椎T3',
  vertebrae_T2: '胸椎T2',
  vertebrae_T1: '胸椎T1',
  vertebrae_C7: '颈椎C7',
  vertebrae_C6: '颈椎C6',
  vertebrae_C5: '颈椎C5',
  vertebrae_C4: '颈椎C4',
  vertebrae_C3: '颈椎C3',
  vertebrae_C2: '颈椎C2',
  vertebrae_C1: '颈椎C1',
  vertebrae: '椎骨',
  intervertebral_discs: '椎间盘',
  spinal_cord: '脊髓',
  spinal_canal: '椎管',

  // 肋骨
  rib_left_1: '左第1肋',
  rib_left_2: '左第2肋',
  rib_left_3: '左第3肋',
  rib_left_4: '左第4肋',
  rib_left_5: '左第5肋',
  rib_left_6: '左第6肋',
  rib_left_7: '左第7肋',
  rib_left_8: '左第8肋',
  rib_left_9: '左第9肋',
  rib_left_10: '左第10肋',
  rib_left_11: '左第11肋',
  rib_left_12: '左第12肋',
  rib_right_1: '右第1肋',
  rib_right_2: '右第2肋',
  rib_right_3: '右第3肋',
  rib_right_4: '右第4肋',
  rib_right_5: '右第5肋',
  rib_right_6: '右第6肋',
  rib_right_7: '右第7肋',
  rib_right_8: '右第8肋',
  rib_right_9: '右第9肋',
  rib_right_10: '右第10肋',
  rib_right_11: '右第11肋',
  rib_right_12: '右第12肋',

  // 肌肉
  gluteus_maximus_left: '左臀大肌',
  gluteus_maximus_right: '右臀大肌',
  gluteus_medius_left: '左臀中肌',
  gluteus_medius_right: '右臀中肌',
  gluteus_minimus_left: '左臀小肌',
  gluteus_minimus_right: '右臀小肌',
  autochthon_left: '左竖脊肌',
  autochthon_right: '右竖脊肌',
  iliopsoas_left: '左髂腰肌',
  iliopsoas_right: '右髂腰肌',

  // 腺体
  adrenal_gland_right: '右肾上腺',
  adrenal_gland_left: '左肾上腺',
  thyroid_gland: '甲状腺',

  // 脑部
  brain: '脑',

  // 面部结构
  face: '面部',
  eyeball_left: '左眼球',
  eyeball_right: '右眼球',
  lens_left: '左晶状体',
  lens_right: '右晶状体',
  optic_nerve_left: '左视神经',
  optic_nerve_right: '右视神经',
  lacrimal_gland_left: '左泪腺',
  lacrimal_gland_right: '右泪腺',

  // 其他结构
  body_trunc: '躯干',
  body_extremities: '四肢',
  skin: '皮肤',
  breast_left: '左乳房',
  breast_right: '右乳房',
};

// 任务名称中文映射
export const TASK_NAMES_CN: Record<string, string> = {
  total: '全身分割',
  total_v1: '全身分割 V1',
  total_mr: '全身分割 (MRI)',
  lung_vessels: '肺血管',
  body: '身体轮廓',
  cerebral_bleed: '脑出血',
  hip_implant: '髋关节假体',
  coronary_arteries: '冠状动脉',
  pleural_pericard_effusion: '胸腔/心包积液',
  liver_vessels: '肝血管',
  heartchambers_highres: '心腔高分辨率',
  head_glands_cavities: '头部腺体和腔隙',
  head_muscles: '头部肌肉',
  headneck_bones_vessels: '头颈骨和血管',
  headneck_muscles: '头颈肌肉',
  oculomotor_muscles: '眼外肌',
  lung_nodules: '肺结节',
  kidney_cysts: '肾囊肿',
  breasts: '乳房',
  ventricle_parts: '脑室分区',
  liver_segments: '肝分段',
  liver_segments_mr: '肝分段 (MRI)',
  craniofacial_structures: '颅面结构',
  face: '面部',
  face_mr: '面部 (MRI)',
  brain_structures: '脑结构',
  thigh_shoulder_muscles: '大腿肩部肌肉',
  appendicular_bones: '四肢骨',
  tissue_types: '组织类型',
  vertebrae_body: '椎体',
  test: '测试',
};

export function getOrganDisplayName(organ: string): string {
  return ORGAN_NAMES_CN[organ] || organ.replace(/_/g, ' ');
}

export function getTaskDisplayName(task: string): string {
  return TASK_NAMES_CN[task] || task.replace(/_/g, ' ');
}

