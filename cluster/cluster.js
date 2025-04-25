const plotArea = document.getElementById('plotArea');
let points = [];

function drawPoint(p, color='white') {
    const dot = document.createElement('div');
    dot.style.position = 'absolute';
    dot.style.width = '14px';
    dot.style.height = '14px';
    dot.style.borderRadius = '50%';
    dot.style.background = color;
    dot.style.left = (p.x - 7) + 'px';
    dot.style.top = (p.y - 7) + 'px';
    plotArea.appendChild(dot);
}

plotArea.addEventListener('click', (e) => {
    const r = plotArea.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    points.push({ x, y });
    drawPoint({ x, y });
});

document.getElementById('decreaseBtn').addEventListener('click', () => {
    const input = document.getElementById('kCount');
    let v = Math.max(1, parseInt(input.value) - 1);
    input.value = v;
});
document.getElementById('increaseBtn').addEventListener('click', () => {
    const input = document.getElementById('kCount');
    let v = parseInt(input.value) + 1;
    input.value = v;
});

document.getElementById('clearBtn').addEventListener('click', () => {
    points = [];
    plotArea.innerHTML = '';
});

document.getElementById('clusterBtn').addEventListener('click', () => {
    const k = Math.max(1, parseInt(document.getElementById('kCount').value));
    fetch('cluster.php', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k, points })
    })
        .then(res => res.json())
        .then(json => {
            plotArea.innerHTML = '';
            const colors = ['#3BA0FF','#FF8C00','#8A2BE2','#00FA9A','#FFD700','#FF1493'];
            json.clusters.forEach((cluster, idx) => {
                const color = colors[idx % colors.length];
                cluster.forEach(p => drawPoint(p, color));
            });
        })
        .catch(err => console.error('Ошибка:', err));
});