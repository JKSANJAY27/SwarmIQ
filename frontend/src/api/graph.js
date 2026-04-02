import service, { requestWithRetry } from './index'

/**
 * Fast Build Graph (SwarmIQ Backend)
 * @param {FormData} formData - 包含files, goal 等
 * @returns {Promise}
 */
export function buildGraphFast(formData) {
  return requestWithRetry(() => 
    service({
      url: '/api/graph/build',
      method: 'post',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  )
}

/**
 * 构建图谱
 * @param {Object} data - 包含project_id, graph_name等
 * @returns {Promise}
 */
export function buildGraph(data) {
  return requestWithRetry(() =>
    service({
      url: '/api/graph/build',
      method: 'post',
      data
    })
  )
}

/**
 * 查询任务状态 (Mocked for SwarmIQ)
 */
export function getTaskStatus(taskId) {
  return Promise.resolve({ success: true, data: { status: 'completed' } })
}

/**
 * 获取图谱数据 (Mocked for SwarmIQ)
 */
export function getGraphData(graphId) {
  return Promise.resolve({
    success: true,
    data: {
      nodes: [
        { id: '1', label: 'Simulation Environment', group: 1, size: 25 },
        { id: '2', label: 'Agent Personas', group: 2, size: 20 },
        { id: '3', label: 'Events', group: 3, size: 15 },
        { id: '4', label: 'Documents', group: 4, size: 15 }
      ],
      edges: [
        { from: '1', to: '2', label: 'contains' },
        { from: '1', to: '3', label: 'manages' },
        { from: '4', to: '1', label: 'context' }
      ]
    }
  })
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
