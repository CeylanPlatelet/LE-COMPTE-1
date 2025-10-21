// 全局变量
let earth;
const API_BASE = 'http://localhost:5080/api';

// 初始化地球
async function initEarth() {
    earth = new WE.map('earth_div', {
        center: [31.5, 104.0],
        zoom: 7,
        dragging: true,
        scrollWheelZoom: true
    });

    WE.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(earth);

    // 添加标记点
    addMarker(31.6061, 104.3556, "一把刀");
    addMarker(31.6349, 104.1022, "大光包");
    addMarker(30.4889, 102.2689, "野牛沟");

    // 自动旋转效果
    let before = null;
    requestAnimationFrame(function animate(now) {
        if (!before) before = now;
        earth.setCenter([31.5, ((now - before) / 100) % 360 - 180]);
        before = now;
        requestAnimationFrame(animate);
    });
}

// 添加标记点到地球
function addMarker(lat, lon, name) {
    const marker = WE.marker([lat, lon]).addTo(earth);
    marker.bindPopup(`<b>${name}</b><br>经度: ${lon}<br>纬度: ${lat}`, {
        maxWidth: 150,
        closeButton: true
    });
}

// 显示所有案例标记
function showAllCaseMarkers() {
    // 清除现有标记
    earth.markerLayer.clearLayers();

    // 为每个案例添加标记
    caseData.forEach(case_ => {
        if (case_.latitude && case_.longitude) {
            addMarker(case_.latitude, case_.longitude, case_.name || `案例 ${case_.id}`);
        }
    });
}

// 加载案例地图
function loadCaseMap(caseId) {
    const currentCase = caseData.find(c => c.id === caseId);
    if (currentCase && currentCase.latitude && currentCase.longitude) {
        earth.setCenter([currentCase.latitude, currentCase.longitude]);
        earth.setZoom(8);
        
        // 清除现有标记并添加新标记
        earth.markerLayer.clearLayers();
        addMarker(currentCase.latitude, currentCase.longitude, currentCase.name || `案例 ${currentCase.id}`);
    }
}

// 地理信息相关功能
async function loadGeographicData() {
    try {
        const response = await fetch(`${API_BASE}/geographic-data`);
        if (response.ok) {
            const data = await response.json();
            updateGeographicInfo(data);
        } else {
            console.error('Failed to load geographic data');
            loadSampleGeographicData();
        }
    } catch (error) {
        console.error('Error loading geographic data:', error);
        loadSampleGeographicData();
    }
}

// 更新地理信息显示
function updateGeographicInfo(data) {
    const container = document.getElementById('geographicInfo');
    if (!container) return;

    container.innerHTML = `
        <div class="info-group">
            <h3>地形特征</h3>
            <p>海拔: ${data.elevation || '未知'} 米</p>
            <p>坡度: ${data.slope || '未知'} 度</p>
            <p>坡向: ${data.aspect || '未知'}</p>
        </div>
        <div class="info-group">
            <h3>地质条件</h3>
            <p>岩性: ${data.rockType || '未知'}</p>
            <p>土壤类型: ${data.soilType || '未知'}</p>
            <p>地质构造: ${data.geologicalStructure || '未知'}</p>
        </div>
    `;
}

// 加载示例地理数据
function loadSampleGeographicData() {
    const sampleData = {
        elevation: 2500,
        slope: 35,
        aspect: '东南',
        rockType: '砂岩',
        soilType: '山地棕壤',
        geologicalStructure: '断层'
    };
    updateGeographicInfo(sampleData);
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initEarth();
    loadGeographicData();
});