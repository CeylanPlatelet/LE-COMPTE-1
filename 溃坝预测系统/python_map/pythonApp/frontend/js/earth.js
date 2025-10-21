// 3D地球相关代码
let scene, camera, renderer, earth;
let markers = [];
let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };

// 初始化3D地球
function initEarth() {
    // 获取容器
    const container = document.getElementById('globe-container');
    
    // 创建场景
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    // 创建相机
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    camera.position.z = 4;
    
    // 创建渲染器
    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('earth-canvas'),
        antialias: true,
        alpha: true
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    
    // 创建地球（使用苹果地图风格的卫星纹理）
    const geometry = new THREE.SphereGeometry(2, 128, 128);
    
    // 创建纹理加载器
    const textureLoader = new THREE.TextureLoader();
    
    // 使用多个备用地球纹理源，确保加载成功
    const textureUrls = [
        'https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg',
        'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg',
        'https://cdn.jsdelivr.net/gh/mrdoob/three.js@dev/examples/textures/planets/earth_atmos_2048.jpg',
        'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjU2IiBoZWlnaHQ9IjEyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImEiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiM0YTkwZTIiLz48c3RvcCBvZmZzZXQ9IjMwJSIgc3RvcC1jb2xvcj0iIzJjNWFhMCIvPjxzdG9wIG9mZnNldD0iNzAlIiBzdG9wLWNvbG9yPSIjNGE5MGUyIi8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMmM1YWEwIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0idXJsKCNhKSIvPjwvc3ZnPg=='
    ];
    
    let currentTextureIndex3 = 0;
    let earthTexture3 = null;
    
    function loadEarthTexture3() {
        if (currentTextureIndex3 >= textureUrls.length) {
            console.error('所有地球纹理源都加载失败，使用默认颜色');
            // 使用渐变色作为最终备用方案
            earth.material = new THREE.MeshPhongMaterial({
                color: 0x4a90e2,
                shininess: 60,
                specular: 0x2c5aa0
            });
            return;
        }
        
        const url = textureUrls[currentTextureIndex3];
        console.log(`尝试加载地球纹理3 ${currentTextureIndex3 + 1}/${textureUrls.length}: ${url}`);
        
        earthTexture3 = textureLoader.load(
            url,
            function(texture) {
                console.log(`地球纹理3加载成功: ${url}`);
                // 设置纹理参数以获得最佳质量
                texture.wrapS = THREE.RepeatWrapping;
                texture.wrapT = THREE.RepeatWrapping;
                texture.minFilter = THREE.LinearMipmapLinearFilter;
                texture.magFilter = THREE.LinearFilter;
                texture.generateMipmaps = true;
                
                // 更新地球材质
                if (earth && earth.material) {
                    earth.material.map = texture;
                    earth.material.needsUpdate = true;
                    console.log('地球材质3已更新为纹理');
                }
            },
            function(progress) {
                console.log(`纹理3加载进度: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
            },
            function(error) {
                console.warn(`纹理3加载失败: ${url}`, error);
                currentTextureIndex3++;
                setTimeout(loadEarthTexture3, 1000); // 1秒后尝试下一个纹理
            }
        );
    }
    
    // 创建地球材质（先使用基础颜色，纹理加载成功后会自动更新）
    const material = new THREE.MeshPhongMaterial({
        color: 0x4a90e2,         // 基础地球蓝色
        shininess: 60,           // 适中的光泽度
        specular: 0x111111,      // 低强度高光，更自然
        transparent: false,
        side: THREE.FrontSide
    });
    
    earth = new THREE.Mesh(geometry, material);
    earth.rotation.y = Math.PI;
    scene.add(earth);
    
    // 开始加载纹理
    loadEarthTexture3();
    
    // 添加环境光和点光源 - 调整光照增强对比度
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);  // 降低环境光
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 2.5);  // 增强光源强度
    pointLight.position.set(5, 3, 5);
    scene.add(pointLight);
    
    // 添加第二个光源增强对比度
    const pointLight2 = new THREE.PointLight(0xffffff, 1.5);
    pointLight2.position.set(-3, -2, 4);
    scene.add(pointLight2);
    
    // 添加事件监听
    const canvas = renderer.domElement;
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onMouseWheel);
    
    // 自动旋转
    autoRotate();
    
    // 响应窗口大小变化
    window.addEventListener('resize', onWindowResize);

    // 立即添加初始标记点
    setTimeout(() => {
        addMarker(30.9810, 102.0250, 0xff0000, '丹巴梅龙沟');
        addMarker(29.5909, 102.1797, 0x00ff00, '泸定县烂田湾');
        addMarker(31.8467, 104.4272, 0xffff00, '唐家山堰塞坝');
        console.log('初始标记点已添加');
    }, 1000);
}

// 创建文字贴图
function createTextTexture(text) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 64;
    
    ctx.fillStyle = 'white';
    ctx.font = 'bold 32px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
}

// 添加标记点
function addMarker(lat, lon, color = 0xff0000, name = '') {
    console.log('添加标记点:', name, lat, lon);
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    
    // 创建标记组
    const markerGroup = new THREE.Group();
    
    // 创建主标记球体
    const geometry = new THREE.SphereGeometry(0.03, 16, 16);
    const material = new THREE.MeshBasicMaterial({ 
        color: color,
        transparent: true,
        opacity: 0.9
    });
    const marker = new THREE.Mesh(geometry, material);
    
    // 创建发光效果
    const glowGeometry = new THREE.SphereGeometry(0.05, 16, 16);
    const glowMaterial = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.3
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    
    // 创建标记文字 (已隐藏)
    // const sprite = new THREE.Sprite(
    //     new THREE.SpriteMaterial({
    //         map: createTextTexture(name),
    //         sizeAttenuation: false
    //     })
    // );
    // sprite.scale.set(0.2, 0.2, 1);
    // sprite.position.y = 0.15;
    
    // 计算位置
    const radius = 2.05; // 稍微高于地球表面
    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.cos(phi);
    const z = radius * Math.sin(phi) * Math.sin(theta);
    
    markerGroup.position.set(x, y, z);
    marker.position.set(0, 0, 0);
    glow.position.set(0, 0, 0);
    
    // 添加到组
    markerGroup.add(marker);
    markerGroup.add(glow);
    // markerGroup.add(sprite); // 文字精灵已隐藏
    
    // 旋转标记组使其面向地球外部
    markerGroup.lookAt(0, 0, 0);
    markerGroup.rotateY(Math.PI);
    
    // 存储标记信息
    markerGroup.userData = { lat, lon, name, color };
    
    // 添加交互事件
    markerGroup.children.forEach(child => {
        child.userData = { lat, lon, name };
    });
    
    markers.push(markerGroup);
    scene.add(markerGroup);
    console.log('标记点已添加到场景');
    
    return markerGroup;
}

// 自动旋转
function autoRotate() {
    if (!isDragging) {
        earth.rotation.y += 0.001;
        
        // 让标记发光效果脉动
        markers.forEach(marker => {
            if (marker.children[1]) { // glow effect
                const time = Date.now() * 0.005;
                marker.children[1].material.opacity = 0.2 + 0.2 * Math.sin(time);
            }
        });
    }
    renderer.render(scene, camera);
    requestAnimationFrame(autoRotate);
}

// 鼠标事件处理
function onMouseDown(event) {
    // 检查是否点击了标记
    const rect = renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);
    
    // 检测交互
    const intersects = raycaster.intersectObjects(markers.flatMap(group => group.children), true);
    
    if (intersects.length > 0) {
        const selectedObject = intersects[0].object;
        const markerData = selectedObject.parent.userData;
        
        // 显示位置信息
        const locationInfo = document.getElementById('location-info');
        const longitudeDisplay = document.getElementById('longitude-display');
        const latitudeDisplay = document.getElementById('latitude-display');
        
        if (locationInfo && longitudeDisplay && latitudeDisplay) {
            locationInfo.classList.remove('hidden');
            longitudeDisplay.textContent = markerData.lon.toFixed(4);
            latitudeDisplay.textContent = markerData.lat.toFixed(4);
            
            // 添加地点名称
            const nameElement = document.getElementById('location-name') || 
                document.createElement('p');
            nameElement.id = 'location-name';
            nameElement.className = 'text-white text-sm';
            nameElement.textContent = `地点: ${markerData.name}`;
            
            if (!document.getElementById('location-name')) {
                locationInfo.insertBefore(nameElement, locationInfo.firstChild);
            }
        }
        
        // 高亮显示选中的标记
        markers.forEach(group => {
            group.children.forEach(child => {
                if (child.material) {
                    child.material.opacity = 0.3;
                }
            });
        });
        
        selectedObject.parent.children.forEach(child => {
            if (child.material) {
                child.material.opacity = 1;
            }
        });
        
        // 阻止地球拖拽
        event.stopPropagation();
        return;
    } else {
        // 点击空白处时隐藏信息框并重置标记透明度
        const locationInfo = document.getElementById('location-info');
        if (locationInfo) {
            locationInfo.classList.add('hidden');
        }
        
        markers.forEach(group => {
            group.children.forEach(child => {
                if (child.material) {
                    child.material.opacity = child instanceof THREE.Sprite ? 1 : 0.9;
                }
            });
        });
        
        // 设置拖拽标志
        isDragging = true;
        previousMousePosition = {
            x: event.clientX,
            y: event.clientY
        };
    }
}

function onMouseMove(event) {
    if (!isDragging) return;
    
    const deltaMove = {
        x: event.clientX - previousMousePosition.x,
        y: event.clientY - previousMousePosition.y
    };
    
    earth.rotation.y += deltaMove.x * 0.005;
    earth.rotation.x += deltaMove.y * 0.005;
    
    previousMousePosition = {
        x: event.clientX,
        y: event.clientY
    };
}

function onMouseUp() {
    isDragging = false;
}

function onMouseWheel(event) {
    camera.position.z = Math.max(3, Math.min(8, camera.position.z + event.deltaY * 0.01));
}

function onWindowResize() {
    const container = document.getElementById('globe-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}