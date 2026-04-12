/* ─────────────────────────────────────────
   FALCON — script.js  (backend-connected)
   ───────────────────────────────────────── */

const SEG_DURATION = 0.7;   // animation seconds per step

let currentTab    = 'text';
let videoUploaded = false;
let queryReady    = false;
let falconDone    = false;
let linearDone    = false;
let lastResult    = null;   // full JSON from /analyze

/* ── File upload: video ── */
document.getElementById('videoInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById('videoCard').classList.add('uploaded');
    document.getElementById('videoHint').classList.add('hidden');
    document.getElementById('videoNameText').textContent = file.name;
    document.getElementById('videoName').classList.remove('hidden');
    videoUploaded = true;
    setTimeout(() => {
        const s2 = document.getElementById('s2');
        s2.classList.remove('hidden');
        s2.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 350);
});

/* ── File upload: reference image ── */
document.getElementById('imageInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById('imgCard').classList.add('uploaded');
    document.getElementById('imgHint').classList.add('hidden');
    document.getElementById('imgNameText').textContent = file.name;
    document.getElementById('imgName').classList.remove('hidden');
    queryReady = true;
});

/* ── Text query readiness ── */
document.getElementById('textQuery').addEventListener('input', function (e) {
    queryReady = e.target.value.trim().length > 0;
});

/* ── Tab switching ── */
function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    if (tab === 'text') {
        document.getElementById('textTab').classList.remove('hidden');
        document.getElementById('imageTab').classList.add('hidden');
        queryReady = document.getElementById('textQuery').value.trim().length > 0;
    } else {
        document.getElementById('textTab').classList.add('hidden');
        document.getElementById('imageTab').classList.remove('hidden');
        queryReady = document.getElementById('imageInput').files.length > 0;
    }
}

/* ── Build segment bars with real counts ── */
function buildBars(numSegs, targetSeg) {
    ['falconBar', 'linearBar'].forEach(id => {
        const bar = document.getElementById(id);
        bar.innerHTML = '';
        for (let i = 0; i < numSegs; i++) {
            const seg = document.createElement('div');
            seg.className = 'seg' + (i === targetSeg ? ' target-mark' : '');
            seg.dataset.i = i;
            bar.appendChild(seg);
        }
    });
}

/* ── Placeholder scanning animation while backend runs ── */
let _scannerTimer = null;

function startScannerAnimation(numSegs) {
    buildBars(numSegs, -1);   // no target mark yet
    let cur = 0;
    const falconSegs = Array.from(document.querySelectorAll('#falconBar .seg'));
    const linearSegs = Array.from(document.querySelectorAll('#linearBar .seg'));

    function scan() {
        falconSegs.forEach(s => s.classList.remove('current'));
        linearSegs.forEach(s => s.classList.remove('current'));
        if (falconSegs[cur]) falconSegs[cur].classList.add('current');
        if (linearSegs[cur]) linearSegs[cur].classList.add('current');
        cur = (cur + 1) % numSegs;
        _scannerTimer = setTimeout(scan, 80);
    }
    scan();
}

function stopScannerAnimation() {
    if (_scannerTimer) { clearTimeout(_scannerTimer); _scannerTimer = null; }
    // clear current highlights
    document.querySelectorAll('.seg.current').forEach(s => s.classList.remove('current'));
}

/* ── Start processing — upload to backend ── */
async function startProcessing() {
    if (!videoUploaded) { alert('Please upload a video first.'); return; }
    if (!queryReady)    { alert('Please provide a query.'); return; }

    const btn = document.getElementById('processBtn');
    btn.disabled    = true;
    btn.textContent = 'UPLOADING…';

    // Fade s1 + s2 out
    ['s1', 's2'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.add('fade-out');
        setTimeout(() => el.classList.add('hidden'), 380);
    });

    // Show s3 immediately with scanning animation
    await new Promise(r => setTimeout(r, 420));
    const s3 = document.getElementById('s3');
    s3.classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    startScannerAnimation(30);   // placeholder 30 segs while waiting

    document.getElementById('falconStatus').textContent =
        'RL agent scanning video segments…';
    document.getElementById('linearStatus').textContent =
        'Linear baseline preparing sequential scan…';

    // Build request
    const formData = new FormData();
    formData.append('video', document.getElementById('videoInput').files[0]);
    if (currentTab === 'text') {
        formData.append('query', document.getElementById('textQuery').value.trim());
    }

    try {
        const resp = await fetch('/analyze', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.error || `Server error ${resp.status}`);
        }

        lastResult = await resp.json();
        falconDone = false;
        linearDone = false;

        stopScannerAnimation();

        const { num_segments, target_segment, visited_segments,
                timestamp, mode } = lastResult;

        buildBars(num_segments, target_segment);

        const modeLabel = mode === 'sliding_window' ? 'SLIDING WINDOW' : 'DIRECT';
        document.getElementById('falconStatus').textContent =
            `Inference complete [${modeLabel}]. Replaying RL agent path…`;

        setTimeout(() => {
            runFalcon(visited_segments, num_segments, target_segment, timestamp);
            runLinear(num_segments, target_segment, timestamp);
        }, 600);

    } catch (err) {
        stopScannerAnimation();
        document.getElementById('falconStatus').textContent =
            `Error: ${err.message}`;
        document.getElementById('linearStatus').textContent =
            'Analysis failed — check the console.';
        console.error(err);
        btn.disabled    = false;
        btn.textContent = 'START PROCESSING';
    }
}

/* ─────────────────────────────────────────
   FALCON animation — replays real RL path
   ───────────────────────────────────────── */
function runFalcon(path, numSegs, targetSeg, realTimestamp) {
    const segs = Array.from(document.querySelectorAll('#falconBar .seg'));
    const vis  = new Set();

    const statusMsgs = [
        'Computing low-res features for all segments…',
        'RL policy selecting next high-value segment…',
        'Extracting hi-res features from selected segment…',
        'Running temporal state propagation…',
        'Computing updated cosine similarities…',
        'Evaluating reward signal — policy converging…',
    ];

    let step = 0;
    document.getElementById('falconStatus').textContent = 'Initializing RL agent…';

    function tick() {
        if (step >= path.length) {
            segs.forEach(s => s.classList.remove('current'));
            if (segs[targetSeg]) {
                segs[targetSeg].classList.remove('target-mark');
                segs[targetSeg].classList.add('found', 'current');
            }

            const pct = Math.round((vis.size / numSegs) * 100);
            document.getElementById('falconPct').textContent = pct + '%';

            const conf = lastResult ? lastResult.confidence : 0;
            const pt   = lastResult ? lastResult.processing_time : 0;
            document.getElementById('rFalconTime').textContent    = pt + 's';
            document.getElementById('rFalconVisited').textContent = pct + '%';

            const status = document.getElementById('falconStatus');
            status.textContent = `Target located. Confidence: ${(conf * 100).toFixed(1)}%`;
            status.classList.add('done');

            document.getElementById('falconTSVal').textContent = realTimestamp;
            document.getElementById('falconTS').classList.add('visible');

            falconDone = true;
            checkBothDone();
            return;
        }

        const idx = path[step];
        segs.forEach(s => s.classList.remove('current'));
        if (segs[idx]) {
            segs[idx].classList.add('visited', 'current');
            if (segs[idx].classList.contains('target-mark') && idx !== targetSeg)
                segs[idx].classList.remove('target-mark');
        }
        vis.add(idx);

        const pct = Math.round((vis.size / numSegs) * 100);
        document.getElementById('falconPct').textContent = pct + '%';
        document.getElementById('falconStatus').textContent =
            statusMsgs[step % statusMsgs.length];

        step++;
        setTimeout(tick, SEG_DURATION * 1000);
    }

    tick();
}

/* ─────────────────────────────────────────
   Linear animation — scans ALL segments
   sequentially, marks found at targetSeg
   but keeps going to show full cost
   ───────────────────────────────────────── */
function runLinear(numSegs, targetSeg, realTimestamp) {
    const segs = Array.from(document.querySelectorAll('#linearBar .seg'));
    let cur = 0;
    let foundAt = -1;

    document.getElementById('linearStatus').textContent =
        'Processing frames sequentially from start…';

    const msgs = [
        i => `Extracting features from segment ${i + 1}…`,
        i => `Computing similarity score for segment ${i + 1}…`,
        i => `No match — advancing to segment ${i + 2}…`,
    ];

    function tick() {
        if (cur >= numSegs) {
            // finished scanning everything
            linearDone = true;
            checkBothDone();
            return;
        }

        if (cur > 0 && segs[cur - 1]) segs[cur - 1].classList.remove('current');
        if (segs[cur]) {
            segs[cur].classList.add('visited', 'current');
            if (segs[cur].classList.contains('target-mark') && cur !== targetSeg)
                segs[cur].classList.remove('target-mark');
        }

        const pct = Math.round(((cur + 1) / numSegs) * 100);
        document.getElementById('linearPct').textContent = pct + '%';

        if (cur === targetSeg && foundAt === -1) {
            foundAt = cur;
            if (segs[cur]) {
                segs[cur].classList.remove('target-mark');
                segs[cur].classList.add('found');
            }
            document.getElementById('linearStatus').textContent =
                `Match found at seg ${targetSeg + 1} — still scanning remaining segments…`;
            document.getElementById('linearTSVal').textContent = realTimestamp;
            document.getElementById('linearTS').classList.add('visible');
        } else if (foundAt === -1) {
            document.getElementById('linearStatus').textContent = msgs[cur % 3](cur);
        }

        cur++;

        // slow down slightly after finding, to emphasise wasted effort
        const delay = (foundAt !== -1 && cur > foundAt) ? SEG_DURATION * 600 : SEG_DURATION * 1000;

        if (cur === numSegs) {
            // wrap up
            const status = document.getElementById('linearStatus');
            status.textContent = `Exhaustive scan complete. All ${numSegs} segments processed.`;
            status.classList.add('done');
            document.getElementById('rLinearVisited').textContent = '100%';
            setTimeout(() => { linearDone = true; checkBothDone(); }, delay);
            return;
        }

        setTimeout(tick, delay);
    }

    tick();
}

/* ── Reveal results when both animations finish ── */
function checkBothDone() {
    if (!falconDone || !linearDone) return;

    setTimeout(() => {
        const wrap = document.getElementById('resultsWrap');
        wrap.classList.remove('hidden');

        if (lastResult) {
            const { num_segments, visited_segments, processing_time } = lastResult;

            const falconFrac = visited_segments.length / num_segments;
            const linearFrac = 1.0;   // linear always scans everything

            // Speedup: how many times fewer segments FALCON visited vs linear
            const speedup = falconFrac > 0
                ? (linearFrac / falconFrac).toFixed(1)
                : '—';
            document.getElementById('speedupVal').textContent = speedup + '×';

            // Estimated linear wall-time
            if (processing_time && falconFrac > 0) {
                const estLinear = (processing_time * (linearFrac / falconFrac)).toFixed(1);
                document.getElementById('rLinearTime').textContent = estLinear + 's';
            }

            document.getElementById('rLinearVisited').textContent = '100%';
        }

        setTimeout(() => {
            wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);
    }, 800);
}
