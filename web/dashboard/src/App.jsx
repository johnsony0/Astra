import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000'

function App() {
  const [stats, setStats] = useState(null)
  const [detections, setDetections] = useState([])
  const [devices, setDevices] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchData = async () => {
    try {
      const [statsRes, detectionsRes, devicesRes] = await Promise.all([
        axios.get(`${API_URL}/api/statistics`),
        axios.get(`${API_URL}/api/detections?limit=20`),
        axios.get(`${API_URL}/api/devices`)
      ])
      setStats(statsRes.data)
      setDetections(detectionsRes.data.detections || [])
      setDevices(devicesRes.data.devices || [])
      setLoading(false)
    } catch (error) {
      console.error('Error fetching data:', error)
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="app">
        <div className="loading">Loading...</div>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>Astra</h1>
        <p>Cryptomining Detection Dashboard</p>
      </header>

      <main>
        <section className="stats">
          <div className="stat-card">
            <div className="stat-value">{stats?.total_flows || 0}</div>
            <div className="stat-label">Total Flows</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats?.benign_count || 0}</div>
            <div className="stat-label">Benign</div>
          </div>
          <div className="stat-card stat-card-danger">
            <div className="stat-value">{stats?.malicious_count || 0}</div>
            <div className="stat-label">Malicious</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats?.detection_rate?.toFixed(1) || 0}%</div>
            <div className="stat-label">Detection Rate</div>
          </div>
        </section>

        <section className="devices">
          <h2>Devices ({devices.length})</h2>
          <div className="device-list">
            {devices.map(device => (
              <div key={device.device_id} className="device-card">
                <div className="device-name">{device.name || device.device_id}</div>
                <div className="device-stats">
                  <span>{device.total_flows} flows</span>
                  <span className="device-malicious">{device.malicious_count} malicious</span>
                </div>
                <div className="device-time">{new Date(device.last_seen).toLocaleString()}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="detections">
          <h2>Recent Detections</h2>
          <div className="detection-table">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Device</th>
                  <th>RF</th>
                  <th>SVM</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {detections.map(detection => (
                  <tr key={detection.id} className={detection.label === 1 ? 'malicious' : ''}>
                    <td>{new Date(detection.timestamp).toLocaleTimeString()}</td>
                    <td>{detection.device_id}</td>
                    <td>
                      <span className={detection.rf_prediction === 1 ? 'badge-danger' : 'badge-success'}>
                        {detection.rf_prediction === 1 ? 'Malicious' : 'Benign'}
                      </span>
                      <span className="confidence">{(detection.rf_confidence * 100).toFixed(0)}%</span>
                    </td>
                    <td>
                      <span className={detection.svm_prediction === 1 ? 'badge-danger' : 'badge-success'}>
                        {detection.svm_prediction === 1 ? 'Malicious' : 'Benign'}
                      </span>
                      <span className="confidence">{(detection.svm_confidence * 100).toFixed(0)}%</span>
                    </td>
                    <td>
                      {detection.label === 1 ? (
                        <span className="status-danger">⚠️ Alert</span>
                      ) : (
                        <span className="status-success">✓ Normal</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
