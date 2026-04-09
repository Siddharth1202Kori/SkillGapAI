// frontend/script.js

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Icons
    feather.replace();

    // DOM Elements
    const themeToggle = document.getElementById('themeToggle');
    const heroSection = document.getElementById('heroSection');
    const inputSection = document.getElementById('inputSection');
    const loadingSection = document.getElementById('loadingSection');
    const dashboardSection = document.getElementById('dashboardSection');
    const analysisForm = document.getElementById('analysisForm');
    const startBtn = document.getElementById('startBtn');
    const refineBtn = document.getElementById('refineBtn');
    
    // Output Elements
    const jobsList = document.getElementById('jobsList');
    const jobsCount = document.getElementById('jobsCount');
    const demandSkillsGrid = document.getElementById('demandSkillsGrid');
    const markdownContent = document.getElementById('markdownContent');
    const dashQuery = document.getElementById('dashQuery');

    // ─── Theme Toggle ────────────────────────────────────────────────────────
    let isDarkMode = false;
    themeToggle.addEventListener('click', () => {
        isDarkMode = !isDarkMode;
        if (isDarkMode) {
            document.body.classList.remove('light-mode');
            document.body.classList.add('dark-mode');
            themeToggle.innerHTML = '<i data-feather="sun"></i>';
        } else {
            document.body.classList.remove('dark-mode');
            document.body.classList.add('light-mode');
            themeToggle.innerHTML = '<i data-feather="moon"></i>';
        }
        feather.replace();
    });

    // ─── Flow Navigation ─────────────────────────────────────────────────────

    // 1. Landing -> Input Form
    startBtn.addEventListener('click', () => {
        heroSection.classList.add('hidden');
        inputSection.classList.remove('hidden');
        inputSection.style.animation = "fadeUp 0.6s ease-out forwards";
        document.getElementById('jobRole').focus();
    });

    // 2. Refine -> Back to Input
    refineBtn.addEventListener('click', () => {
        dashboardSection.classList.add('hidden');
        inputSection.classList.remove('hidden');
    });

    // 3. Form Submit -> Loading -> Dashboard
    analysisForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const role = document.getElementById('jobRole').value;
        const bg = document.getElementById('background').value;
        if (!role) return;

        // Transition to loader
        inputSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        
        // Dynamic loader text
        const loaderText = document.getElementById('loaderText');
        const stages = [
            "Fetching live job postings for " + role + "...",
            "Chunking text & generating dense embeddings...",
            "Fetching 8 closest semantic matches from ChromaDB...",
            "Pinging Mistral LLM for tailored gap analysis..."
        ];
        
        let step = 0;
        const loaderInterval = setInterval(() => {
            step++;
            if (step < stages.length) loaderText.innerText = stages[step];
        }, 8000); // Step every 8s while waiting for ML process

        try {
            // Step 1: Submit job and get job_id immediately (non-blocking)
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: role, background: bg })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Failed to start pipeline job.');
            }

            const { job_id } = await res.json();

            // Step 2: Poll /api/status/<job_id> every 3 seconds until done
            const RAGData = await pollForResult(job_id, loaderText, stages, loaderInterval);
            clearInterval(loaderInterval);
            showDashboard(role, RAGData);

        } catch (error) {
            clearInterval(loaderInterval);
            alert('Failed to connect to API Backend: ' + error.message);
            loadingSection.classList.add('hidden');
            inputSection.classList.remove('hidden');
        }
    });

    // ─── Polling Helper ──────────────────────────────────────────────────────

    async function pollForResult(jobId, loaderText, stages, loaderInterval) {
        const maxAttempts = 100;  // 100 * 3s = 5 minutes max
        for (let i = 0; i < maxAttempts; i++) {
            await new Promise(r => setTimeout(r, 3000)); // Wait 3s between polls

            const poll = await fetch(`/api/status/${jobId}`);
            if (!poll.ok) throw new Error('Polling failed: server error.');

            const job = await poll.json();

            if (job.status === 'done') return job.result;
            if (job.status === 'error') throw new Error(job.error || 'Pipeline failed.');

            // Update loading message while we wait
            if (loaderText && stages) {
                const stageIdx = Math.min(Math.floor(i / 3), stages.length - 1);
                loaderText.innerText = stages[stageIdx];
            }
        }
        throw new Error('Analysis timed out after 5 minutes. Please try again.');
    }

    // ─── Render Results ──────────────────────────────────────────────────────
    
    function showDashboard(role, data) {
        loadingSection.classList.add('hidden');
        dashboardSection.classList.remove('hidden');
        dashQuery.innerText = `Query: ${data.query}`;

        // 1. Render Matched Jobs
        jobsCount.innerText = data.matched_jobs ? data.matched_jobs.length : 0;
        jobsList.innerHTML = '';
        if (data.matched_jobs) {
            data.matched_jobs.forEach((job, index) => {
                const div = document.createElement('div');
                div.className = 'job-card';
                div.innerHTML = `
                    <div style="font-weight: 600;">${index + 1}. ${job.title} <span style="font-weight: 400; color: var(--text-secondary);">@ ${job.company} — ${job.location || 'Remote'}</span></div>
                    <div style="font-size: 0.8125rem; color: var(--text-secondary); margin-top: 0.25rem;">
                        <strong>Score:</strong> ${job.score.toFixed(3)} | <strong>Skills:</strong> ${job.skills}
                    </div>
                    ${job.url ? `<a href="${job.url}" target="_blank" style="font-size: 0.8125rem; color: var(--primary); margin-top: 0.25rem; display: inline-block;">View Original Job ↗</a>` : ''}
                `;
                jobsList.appendChild(div);
            });
        }

        // 2. Render In-Demand Skills detected
        demandSkillsGrid.innerHTML = '';
        if (data.in_demand_skills) {
            data.in_demand_skills.forEach(skill => {
                const span = document.createElement('span');
                span.className = 'job-skill-tag';
                span.innerText = skill;
                demandSkillsGrid.appendChild(span);
            });
        }

        // 3. Render Mistral LLM Full Markdown Response
        if (data.analysis) {
            // Convert Markdown to clean HTML safely
            markdownContent.innerHTML = marked.parse(data.analysis);
        } else {
            markdownContent.innerHTML = "<p>No analysis generated.</p>";
        }
        
        feather.replace();
    }
});
