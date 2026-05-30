const api = {
  history: '/api/history',
  chat: '/api/chat'
}

const historyList = document.getElementById('historyList')
const promptForm = document.getElementById('promptForm')
const promptInput = document.getElementById('promptInput')
const logsEl = document.getElementById('logs')
const summaryEl = document.getElementById('summary')
const plotsGrid = document.getElementById('plotsGrid')

let pollInterval = null

async function loadHistory(){
  const res = await fetch(api.history)
  const items = await res.json()
  historyList.innerHTML = ''
  items.forEach(it =>{
    const el = document.createElement('div')
    el.className = 'history-item'
    el.textContent = (it.user_query || "(no query)") + ' — ' + (new Date(it.modified*1000)).toLocaleString()
    el.onclick = ()=> loadRun(it.run_id)
    historyList.appendChild(el)
  })
}

async function loadRun(run_id){
  stopPolling()
  const res = await fetch(`${api.history}/${run_id}`)
  if(!res.ok){ alert('Run not found'); return }
  const state = await res.json()
  renderState(state)
}

function renderState(state){
  logsEl.textContent = (state.logs || []).join('\n') || 'No logs yet'
  summaryEl.innerHTML = state.summary_html || '<pre>No summary</pre>'
  plotsGrid.innerHTML = ''
  const arts = state.artifact_paths || {}
  Object.entries(arts).forEach(([k,v])=>{
    if(!v) return
    if(v.endsWith('.png') || v.endsWith('.jpg') || v.endsWith('.jpeg')){
      let href = normalizeToStorageUrl(v)
      const img = document.createElement('img')
      img.src = href
      img.alt = k
      plotsGrid.appendChild(img)
    }else if(v.endsWith('.html')){
      const a = document.createElement('a')
      a.href = normalizeToStorageUrl(v)
      a.target = '_blank'
      a.textContent = 'Open dashboard'
      const wrap = document.createElement('div')
      wrap.appendChild(a)
      plotsGrid.appendChild(wrap)
    }
  })
}

function normalizeToStorageUrl(path){
  if(path.startsWith('/storage')) return path
  const idx = path.indexOf('storage')
  if(idx>=0) return '/' + path.slice(idx)
  return path
}

promptForm.addEventListener('submit', async (e)=>{
  e.preventDefault()
  const q = promptInput.value.trim()
  if(!q) return
  const res = await fetch(api.chat, {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({user_query:q})})
  const data = await res.json()
  if(data.run_id){
    logsEl.textContent = 'Run started: ' + data.run_id
    startPolling(data.run_id)
    await loadHistory()
  }
})

function startPolling(run_id){
  stopPolling()
  pollInterval = setInterval(async ()=>{
    const res = await fetch(`${api.history}/${run_id}`)
    if(!res.ok) return
    const state = await res.json()
    renderState(state)
    if(state.current_stage === 'finished' || state.current_stage === 'error'){
      stopPolling()
    }
  }, 2000)
}

function stopPolling(){
  if(pollInterval) clearInterval(pollInterval)
  pollInterval = null
}

loadHistory()
// Global DOM Elements
const sidebar = document.getElementById('sidebar');
const toggleSidebarBtn = document.getElementById('toggle-sidebar');
const newChatBtn = document.getElementById('new-chat-btn');
const historyList = document.getElementById('history-list');
const messageContainer = document.getElementById('message-container');
const welcomeScreen = document.getElementById('welcome-screen');
const progressPanel = document.getElementById('progress-panel');
const progressLogs = document.getElementById('progress-logs');
const progressBarFill = document.getElementById('progress-bar-fill');
const currentStepLabel = document.getElementById('current-step-label');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const headerTitle = document.getElementById('header-title');

// Modal Elements
const imageModal = document.getElementById('image-modal');
const modalImg = document.getElementById('modal-img');
const modalCaption = document.getElementById('modal-caption');

// Active state tracking
let isExecuting = false;
let progressInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    
    // Sidebar toggle (Mobile support)
    toggleSidebarBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // New Chat Button
    newChatBtn.addEventListener('click', () => {
        resetChat();
    });

    // Form submit event
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        handleSubmit();
    });

    // Auto-grow textarea
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight - 6) + 'px';
    });
});

// Load history list from FastAPI API
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        if (!response.ok) throw new Error('Failed to fetch history');
        
        const history = await response.ok ? await response.json() : [];
        renderHistoryList(history);
    } catch (error) {
        console.error('Error loading history:', error);
        historyList.innerHTML = `<div class="history-empty">Error loading conversations</div>`;
    }
}

// Render history item DOM
function renderHistoryList(history) {
    if (history.length === 0) {
        historyList.innerHTML = `<div class="history-empty">No past runs found</div>`;
        return;
    }

    historyList.innerHTML = '';
    history.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.setAttribute('data-id', item.id);
        div.innerHTML = `
            <i class="fa-regular fa-message"></i>
            <span class="history-item-text" title="${escapeHtml(item.query)}">${escapeHtml(item.query)}</span>
        `;
        
        div.addEventListener('click', () => {
            document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
            div.classList.add('active');
            loadConversation(item.id);
            // Close mobile sidebar if open
            sidebar.classList.remove('open');
        });
        
        historyList.appendChild(div);
    });
}

// Load detail of a single past run
async function loadConversation(id) {
    try {
        welcomeScreen.style.display = 'none';
        progressPanel.style.display = 'none';
        
        // Fetch run details
        const response = await fetch(`/api/history/${id}`);
        if (!response.ok) throw new Error('Failed to fetch run details');
        const data = await response.json();
        
        // Set Header title
        headerTitle.innerText = `Analysis: "${data.query}"`;
        
        // Clear message container
        messageContainer.innerHTML = '';
        
        // Render User Message
        appendMessage('user', data.query);
        
        // Render Agent Response
        appendAgentResponse(data.summary_html, data.plots, data.logs);
        
    } catch (error) {
        console.error('Error loading conversation details:', error);
        messageContainer.innerHTML = `<div class="message error"><div class="message-bubble">Failed to load run details.</div></div>`;
    }
}

// Reset chat workspace to welcome screen
function resetChat() {
    welcomeScreen.style.display = 'block';
    progressPanel.style.display = 'none';
    messageContainer.innerHTML = '';
    welcomeScreen.appendChild(welcomeScreen.querySelector('.welcome-icon').parentNode); // re-append structure if needed
    messageContainer.appendChild(welcomeScreen);
    headerTitle.innerText = "Autonomous Data Analyst Pipeline";
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.focus();
}

// Auto-fill query inputs from suggestions
window.fillInput = function(text) {
    chatInput.value = text;
    chatInput.style.height = 'auto';
    chatInput.style.height = (chatInput.scrollHeight - 6) + 'px';
    chatInput.focus();
};

// Handle submission of a new prompt
async function handleSubmit() {
    if (isExecuting) return;
    
    const query = chatInput.value.trim();
    if (!query) return;
    
    isExecuting = true;
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.disabled = true;
    sendBtn.disabled = true;
    
    // Clear chat display & show user query
    welcomeScreen.style.display = 'none';
    messageContainer.innerHTML = '';
    appendMessage('user', query);
    
    // Show Progress panel
    progressPanel.style.display = 'block';
    progressLogs.innerHTML = '';
    progressBarFill.style.width = '5%';
    currentStepLabel.innerText = "Initializing";
    
    // Start progress simulation logs
    startProgressSimulation(query);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Pipeline execution failed.');
        }
        
        const data = await response.json();
        
        // Finish progress bar
        clearInterval(progressInterval);
        progressBarFill.style.width = '100%';
        currentStepLabel.innerText = "Success";
        appendProgressLog('check', "Full pipeline execution finished successfully!");
        
        // Hide progress panel after a short delay and render response
        setTimeout(() => {
            progressPanel.style.display = 'none';
            appendAgentResponse(data.summary_html, data.plots, data.logs);
            isExecuting = false;
            chatInput.disabled = false;
            sendBtn.disabled = false;
            loadHistory(); // Reload history list to include new item
        }, 1000);
        
    } catch (error) {
        console.error('Pipeline error:', error);
        clearInterval(progressInterval);
        progressPanel.style.display = 'none';
        
        appendMessage('agent', `<div style="color: #ef4444; font-weight: 500;">
            <i class="fa-solid fa-circle-exclamation"></i> Pipeline Execution Error:
            <pre style="margin-top: 8px; font-size: 0.8rem; white-space: pre-wrap; font-family: monospace;">${escapeHtml(error.message)}</pre>
        </div>`);
        
        isExecuting = false;
        chatInput.disabled = false;
        sendBtn.disabled = false;
    }
}

// Append a standard message bubble
function appendMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.innerHTML = `
        <div class="avatar">
            <i class="fa-solid ${sender === 'user' ? 'fa-user' : 'fa-robot'}"></i>
        </div>
        <div class="message-content">
            <span class="message-sender">${sender === 'user' ? 'You' : 'Analyst AI'}</span>
            <div class="message-bubble">${text}</div>
        </div>
    `;
    messageContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Append the sophisticated structured agent response
function appendAgentResponse(summaryHtml, plots, logs) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent';
    
    // Build plot gallery HTML if plots exist
    let plotGalleryHtml = '';
    if (plots && plots.length > 0) {
        plotGalleryHtml = `
            <div class="plot-gallery">
                <h4>Statistical Visualizations (EDA)</h4>
                <div class="plot-grid">
                    ${plots.map((url, i) => `
                        <div class="plot-item" onclick="openImageModal('${url}', 'Plot ${i + 1}')">
                            <img src="${url}" alt="Plot ${i + 1}" loading="lazy">
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    // Build logs accordion HTML if logs exist
    let logsAccordionHtml = '';
    if (logs && logs.length > 0) {
        const uniqueId = `logs-${Date.now()}`;
        logsAccordionHtml = `
            <div class="logs-box">
                <div class="logs-header" onclick="toggleLogs('${uniqueId}')">
                    <span>Show Execution Logs (${logs.length} entries)</span>
                    <i class="fa-solid fa-chevron-down" id="icon-${uniqueId}"></i>
                </div>
                <div class="logs-content" id="${uniqueId}" style="display: none;">
                    ${logs.map(log => `<div>${escapeHtml(log)}</div>`).join('')}
                </div>
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="avatar">
            <i class="fa-solid fa-chart-pie"></i>
        </div>
        <div class="message-content">
            <span class="message-sender">Analyst AI</span>
            <div class="message-bubble">
                ${summaryHtml}
                ${plotGalleryHtml}
                ${logsAccordionHtml}
            </div>
        </div>
    `;
    messageContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Collapsible logs utility
window.toggleLogs = function(id) {
    const logsContent = document.getElementById(id);
    const icon = document.getElementById(`icon-${id}`);
    if (logsContent.style.display === 'none') {
        logsContent.style.display = 'block';
        icon.classList.replace('fa-chevron-down', 'fa-chevron-up');
    } else {
        logsContent.style.display = 'none';
        icon.classList.replace('fa-chevron-up', 'fa-chevron-down');
    }
};

// Simulate step-by-step logs for interactive display
function startProgressSimulation(query) {
    const steps = [
        { label: "Data Collection", log: "Scraping target websites and collecting raw sources...", pct: 15 },
        { label: "Extraction", log: "Extracting structural columns and parsing HTML payloads...", pct: 30 },
        { label: "Data Cleaning", log: "Removing duplicates, standardizing data columns, and profiling...", pct: 50 },
        { label: "Feature Engineering", log: "Extracting date-time components and scaling numeric scales...", pct: 70 },
        { label: "Statistical EDA", log: "Generating outlier distributions and bivariate correlation plots...", pct: 85 },
        { label: "BI Dashboarding", log: "Building Tableau dashboard & Plotly interactive metrics...", pct: 95 }
    ];
    
    let index = 0;
    
    // Initial setup log
    appendProgressLog('notch', "Created execution plan successfully.");
    
    progressInterval = setInterval(() => {
        if (index < steps.length) {
            const step = steps[index];
            currentStepLabel.innerText = step.label;
            progressBarFill.style.width = `${step.pct}%`;
            
            // Mark previous as checked
            const lastLogLines = progressLogs.querySelectorAll('.progress-log-line');
            if (lastLogLines.length > 0) {
                const icon = lastLogLines[lastLogLines.length - 1].querySelector('i');
                if (icon.classList.contains('fa-circle-notch')) {
                    icon.classList.replace('fa-circle-notch', 'fa-circle-check');
                    icon.classList.remove('fa-spin');
                }
            }
            
            appendProgressLog('notch', step.log);
            index++;
        }
    }, 2800);
}

function appendProgressLog(iconType, text) {
    const line = document.createElement('div');
    line.className = 'progress-log-line';
    
    let iconHtml = '';
    if (iconType === 'check') {
        iconHtml = `<i class="fa-solid fa-circle-check"></i>`;
    } else if (iconType === 'notch') {
        iconHtml = `<i class="fa-solid fa-circle-notch fa-spin"></i>`;
    }
    
    line.innerHTML = `${iconHtml}<span>${escapeHtml(text)}</span>`;
    progressLogs.appendChild(line);
    progressLogs.scrollTop = progressLogs.scrollHeight;
}

// Fullscreen Image Modal
window.openImageModal = function(src, alt) {
    imageModal.style.display = "block";
    modalImg.src = src;
    modalCaption.innerHTML = alt;
};

window.closeImageModal = function() {
    imageModal.style.display = "none";
};

// Utilities
function scrollToBottom() {
    messageContainer.scrollTop = messageContainer.scrollHeight;
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
