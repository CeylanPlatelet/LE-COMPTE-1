// Global variables
let caseData = [];
let damChart = null;
const API_BASE = 'http://localhost:5080/api';

// Initialize the application
document.addEventListener('DOMContentLoaded', async function() {
    initializeCharts();
    await loadCaseData();
    await initEarth();
    showAllCaseMarkers();
});

// Initialize charts
function initializeCharts() {
    const damCtx = document.getElementById('damChart').getContext('2d');
    damChart = new Chart(damCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '大坝监测曲线',
                data: [],
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#87ceeb'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#87ceeb'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff'
                    }
                }
            }
        }
    });
}

// Load case data
async function loadCaseData() {
    try {
        const response = await fetch(`${API_BASE}/cases`);
        if (response.ok) {
            caseData = await response.json();
            updateCaseList();
            if (caseData.length > 0) {
                updateCaseInfo(caseData[0]);
                loadCaseMap(caseData[0].id);
                await updateDamChart(caseData[0].id);
            }
        } else {
            console.error('Failed to load case data');
            loadSampleData();
        }
    } catch (error) {
        console.error('Error loading case data:', error);
        loadSampleData();
    }
}

// Update case list
function updateCaseList() {
    const caseListContainer = document.getElementById('caseList');
    caseListContainer.innerHTML = '';

    const fields = [
        { key: 'caseRow', label: '案例行' },
        { key: 'slope', label: '坡度' },
        { key: 'slopeLength', label: '坡面长' },
        { key: 'town', label: '城镇' },
        { key: 'date', label: '发生日' },
        { key: 'width', label: '坡宽' },
        { key: 'drainageLength', label: '汇水长' },
        { key: 'townPoint', label: '城镇点' },
        { key: 'basinE', label: '流域E' },
        { key: 'slopeLength2', label: '坡长' },
        { key: 'upstream', label: '上游' },
        { key: 'townPeople', label: '城镇人' },
        { key: 'caseI', label: '案例I' },
        { key: 'vegetationI', label: '植被I' },
        { key: 'overflow', label: '漫流' },
        { key: 'economy', label: '经济' }
    ];

    fields.forEach(field => {
        const caseItem = document.createElement('div');
        caseItem.className = 'case-item';
        caseItem.innerHTML = `
            <label>${field.label}:</label>
            <input type="text" value="${caseData[0][field.key] || 1}" onchange="updateCaseData('${field.key}', this.value)">
        `;
        caseListContainer.appendChild(caseItem);
    });
}

// Update case information
function updateCaseInfo(caseItem) {
    const caseInfoContainer = document.getElementById('caseInfo');
    caseInfoContainer.innerHTML = '';

    const infoGroups = [
        [
            { label: '案例行', value: caseItem.id || 1 },
            { label: '坡度', value: caseItem.slope || 1 },
            { label: '坡面长', value: caseItem.slopeLength || 1 },
            { label: '城镇', value: caseItem.town || 11 }
        ],
        [
            { label: '发生日', value: caseItem.date || 1 },
            { label: '坡宽', value: caseItem.width || 1 },
            { label: '汇水长', value: caseItem.drainageLength || 11 },
            { label: '城镇点', value: caseItem.townPoint || 1 }
        ],
        [
            { label: '流域E', value: caseItem.basinE || 1 },
            { label: '坡长', value: caseItem.slopeLength2 || 1 },
            { label: '上游', value: caseItem.upstream || 11 },
            { label: '城镇人', value: caseItem.townPeople || 1 }
        ],
        [
            { label: '案例I', value: caseItem.caseI || 1 },
            { label: '植被I', value: caseItem.vegetationI || 1 },
            { label: '漫流', value: caseItem.overflow || 11 },
            { label: '经济', value: caseItem.economy || 11 }
        ]
    ];

    infoGroups.forEach(group => {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'flex flex-col space-y-2';
        
        group.forEach(item => {
            const span = document.createElement('span');
            span.textContent = `${item.label}: ${item.value}`;
            groupDiv.appendChild(span);
        });
        
        caseInfoContainer.appendChild(groupDiv);
    });
}

// Update dam monitoring chart
async function updateDamChart(caseId = null) {
    try {
        let damData;
        if (caseId) {
            const response = await fetch(`${API_BASE}/cases/${caseId}/dam-data`);
            if (response.ok) {
                damData = await response.json();
            }
        }
        
        if (!damData) {
            // Generate sample dam monitoring data
            const timeLabels = [];
            const displacementData = [];
            
            for (let i = 0; i < 24; i++) {
                timeLabels.push(`${i}:00`);
                displacementData.push(Math.sin(i * 0.2) * 2 + Math.random() * 0.5 + 2.0);
            }
            damData = { time: timeLabels, displacement: displacementData };
        }

        damChart.data.labels = damData.time;
        damChart.data.datasets[0].data = damData.displacement;
        damChart.update();
    } catch (error) {
        console.error('Error updating dam chart:', error);
        // Fallback to sample data
        const timeLabels = [];
        const displacementData = [];
        
        for (let i = 0; i < 24; i++) {
            timeLabels.push(`${i}:00`);
            displacementData.push(Math.sin(i * 0.2) * 2 + Math.random() * 0.5 + 2.0);
        }

        damChart.data.labels = timeLabels;
        damChart.data.datasets[0].data = displacementData;
        damChart.update();
    }
}

// Case management functions
async function addCase() {
    const newCase = {
        name: `新案例${caseData.length + 1}`,
        longitude: 100.0,
        latitude: 30.0,
        slope: 1,
        slopeLength: 1,
        town: 1,
        date: 1,
        width: 1,
        drainageLength: 1,
        townPoint: 1,
        basinE: 1,
        slopeLength2: 1,
        upstream: 1,
        townPeople: 1,
        caseI: 1,
        vegetationI: 1,
        overflow: 1,
        economy: 1
    };
    
    try {
        const response = await fetch(`${API_BASE}/cases`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(newCase)
        });
        
        if (response.ok) {
            const addedCase = await response.json();
            caseData.push(addedCase);
            updateCaseList();
            updateCaseInfo(addedCase);
            loadCaseMap(addedCase.id);
        } else {
            console.error('Failed to add case');
        }
    } catch (error) {
        console.error('Error adding case:', error);
        // Fallback to local addition
        newCase.id = caseData.length + 1;
        caseData.push(newCase);
        updateCaseList();
        updateCaseInfo(newCase);
        loadCaseMap(newCase.id);
    }
}

async function deleteCase() {
    if (caseData.length > 1) {
        const caseToDelete = caseData[caseData.length - 1];
        
        try {
            const response = await fetch(`${API_BASE}/cases/${caseToDelete.id}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                caseData = caseData.filter(c => c.id !== caseToDelete.id);
                updateCaseList();
                updateCaseInfo(caseData[0]);
            } else {
                console.error('Failed to delete case');
            }
        } catch (error) {
            console.error('Error deleting case:', error);
            // Fallback to local deletion
            caseData.pop();
            updateCaseList();
            updateCaseInfo(caseData[0]);
        }
    }
}

function updateCaseData(field, value) {
    if (caseData.length > 0) {
        caseData[0][field] = value;
        updateCaseInfo(caseData[0]);
        // Refresh map when longitude/latitude change
        if (field === 'longitude' || field === 'latitude') {
            loadCaseMap(caseData[0].id);
        }
    }
}

// Search cases
async function searchCases() {
    const longitude = document.getElementById('longitude').value;
    const caseName = document.getElementById('caseName').value;
    
    try {
        let url = `${API_BASE}/cases?`;
        const params = new URLSearchParams();
        
        if (longitude) {
            const [lon, lat] = longitude.split(',').map(Number);
            if (!isNaN(lon) && !isNaN(lat)) {
                params.append('longitude', lon);
                params.append('latitude', lat);
            }
        }
        
        if (caseName) {
            params.append('name', caseName);
        }
        
        url += params.toString();
        
        const response = await fetch(url);
        if (response.ok) {
            const filteredData = await response.json();
            if (filteredData.length > 0) {
                updateCaseInfo(filteredData[0]);
                await updateFloodChart(filteredData[0].id);
            }
        } else {
            console.error('Search failed');
        }
    } catch (error) {
        console.error('Error searching cases:', error);
    }
}

// Load sample data
function loadSampleData() {
    caseData = [
        { id: 1, name: "一把刀", longitude: 104.3556, latitude: 31.6061, slope: 1, slopeLength: 1, town: 1, date: 1, width: 1, drainageLength: 1, townPoint: 1, basinE: 1, slopeLength2: 1, upstream: 1, townPeople: 1, caseI: 1, vegetationI: 1, overflow: 1, economy: 1 },
        { id: 2, name: "大光包", longitude: 104.1022, latitude: 31.6349, slope: 1, slopeLength: 1, town: 1, date: 1, width: 1, drainageLength: 1, townPoint: 1, basinE: 1, slopeLength2: 1, upstream: 1, townPeople: 1, caseI: 1, vegetationI: 1, overflow: 1, economy: 1 },
        { id: 500, name: "野牛沟", longitude: 102.2689, latitude: 30.4889, slope: 1, slopeLength: 1, town: 1, date: 1, width: 1, drainageLength: 1, townPoint: 1, basinE: 1, slopeLength2: 1, upstream: 1, townPeople: 1, caseI: 1, vegetationI: 1, overflow: 1, economy: 1 }
    ];

    updateCaseList();
    updateCaseInfo(caseData[0]);
    loadCaseMap(caseData[0].id);
    updateFloodChart();
}