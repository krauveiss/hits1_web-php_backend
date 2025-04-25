const form = document.getElementById('dataForm');
form.querySelectorAll('button[data-action]').forEach(btn => {
    btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        const data = new FormData(form);
        data.append('action', action);
        fetch('id3.php', { method: 'POST', body: data })
            .then(res => res.json())
            .then(json => {
                document.getElementById('treeOutput').innerHTML = json.tree;
                document.getElementById('treeResult').style.display = 'block';
                if (action === 'classify') {
                    document.getElementById('classifyOutput').innerHTML = json.result;
                    document.getElementById('classifyResult').style.display = 'block';
                    window.location.href="#classifyOutput";
                } else {
                    document.getElementById('classifyResult').style.display = 'none';
                }
            })
            .catch(err => alert('Ошибка: ' + err));
    });
});