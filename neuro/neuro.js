const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let drawing = false, lastX = 0, lastY = 0;

function initCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.fillStyle = 'white';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.beginPath();
}
initCanvas();

['mousedown', 'touchstart'].forEach(e => canvas.addEventListener(e, startDrawing));
['mousemove', 'touchmove'].forEach(e => canvas.addEventListener(e, draw));
['mouseup', 'mouseout', 'touchend'].forEach(e => canvas.addEventListener(e, endDrawing));

function getCoords(e) {
    const r = canvas.getBoundingClientRect();
    const cx = e.clientX || (e.touches && e.touches[0].clientX);
    const cy = e.clientY || (e.touches && e.touches[0].clientY);
    return {
        x: (cx - r.left) * (canvas.width / r.width),
        y: (cy - r.top) * (canvas.height / r.height)
    };
}
function startDrawing(e) {
    e.preventDefault(); drawing = true;
    const { x, y } = getCoords(e);
    lastX = x; lastY = y;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y);
    ctx.stroke();
}
function draw(e) {
    if (!drawing) return;
    e.preventDefault();
    const { x, y } = getCoords(e);
    ctx.lineTo(x, y);
    ctx.stroke();
}
function endDrawing() {
    drawing = false;
}
function clearCanvas() {
    drawing = false;
    lastX = lastY = 0;
    initCanvas();
    document.getElementById('result').innerHTML = '';
}

function getResizedGridData() {
    const rc = document.createElement('canvas'); rc.width = 50; rc.height = 50;
    const rctx = rc.getContext('2d'); rctx.imageSmoothingEnabled = false;
    rctx.drawImage(canvas, 0, 0, 50, 50);
    const data = rctx.getImageData(0, 0, 50, 50).data;
    const arr = [];
    for (let i = 0; i < data.length; i += 4) arr.push(data[i] / 255);
    return arr;
}
function saveGridToFile() {
    const blob = new Blob([JSON.stringify(getResizedGridData())], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'mnist_grid.json'; a.click();
    URL.revokeObjectURL(url);
}

document.getElementById('submitBtn').addEventListener('click', () => {
    fetch('neuro.php', {
        method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'gridData=' + encodeURIComponent(JSON.stringify(getResizedGridData()))
    })
        .then(res => res.text())
        .then(txt => {
            const resDiv = document.getElementById('result');
            const lines = txt.split(/(?:<br\s*\/?>|\r?\n)+/gi)
                .map(s => s.trim()).filter(s => s);
            const headerLine = lines.shift() || '';
            const details = lines;
            resDiv.innerHTML = `
          <div style="display:inline; align-items:center; justify-content:center; gap:10px;">
            <strong style="font-size:24px;">${headerLine}</strong>
            <button id="toggleBtn" class="btn-toggle">Show details</button>
          </div>
          <div id="details" style="display:none; margin-top:15px; text-align:center; line-height:1.4;">
            ${details.join('<br>')}
          </div>
        `;
            document.getElementById('toggleBtn').onclick = function () {
                const d = document.getElementById('details');
                if (d.style.display === 'none') {
                    d.style.display = 'block'; this.textContent = 'Hide details';
                } else {
                    d.style.display = 'none'; this.textContent = 'Show details';
                }
            };
        }).catch(err => console.error('Ошибка:', err));
});