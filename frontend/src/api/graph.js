import service, { requestWithRetry } from './index'

/**
 * Fast Build Graph (SwarmIQ Backend)
 * @param {FormData} formData - 包含files, goal 等
 * @returns {Promise}
 */
export async function buildGraphFast(formData) {
  // Use pure browser native fetch to absolutely ensure FormData integrity and boundary headers
  // bypassing any global Axios headers that might be silently breaking the multipart upload.
  const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'
  
  const response = await fetch(`${API_URL}/api/graph/build`, {
    method: 'POST',
    body: formData
  })
  
  if (!response.ok) {
    const errText = await response.text()
    throw new Error(`Server returned ${response.status}: ${errText}`)
  }
  
  return await response.json()
}

/**
 * 构建图谱
 * @param {Object} data - 包含project_id, graph_name等
 * @returns {Promise}
 */
export function buildGraph(data) {
  return service({
    url: '/api/graph/build',
    method: 'post',
    data,
    timeout: 0 // Disable timeout completely
  })
}

/**
 * 查询任务状态 (Mocked for SwarmIQ)
 */
export function getTaskStatus(taskId) {
  return Promise.resolve({ success: true, data: { status: 'completed' } })
}

/**
 * 获取图谱数据
 */
export function getGraphData(graphId) {
  return service.get(`/api/simulations/${graphId}/graph`)
}

/**
 * 获取项目信息 (Mocked for SwarmIQ)
 */
export function getProject(projectId) {
  return Promise.resolve({
    success: true,
    data: {
      project_id: projectId,
      graph_id: projectId,
      goal: 'Observe dynamic agent relations within specified scenario boundary'
    }
  })
}
