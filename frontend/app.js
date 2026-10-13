/* =============================================================================
   Parkinson's Multi-Modal ML Dashboard — Client Application Logic (app.js)
   ============================================================================= */

document.addEventListener('DOMContentLoaded', () => {

    const API_BASE = 'http://127.0.0.1:8000';

    // Global State
    let authToken = localStorage.getItem('parkinsons_jwt_token') || null;
    let currentUser = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordedAudioBlob = null;
    let recordTimerInterval = null;
    let recordSeconds = 0;
    let selectedUploadFile = null;
    let useSampleAudioFlag = false;

    // Tremor Model State
    let useTremorSegmentMode = true;
    let selectedTremorUploadFile = null;

    // DOM Elements - Auth & Nav
    const authScreen = document.getElementById('auth-screen');
    const dashboardScreen = document.getElementById('dashboard-screen');
    const navUserControls = document.getElementById('nav-user-controls');
    const userDisplayName = document.getElementById('user-display-name');
    const userRoleTag = document.getElementById('user-role-tag');
    const logoutBtn = document.getElementById('logout-btn');

    // Auth Forms
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const tabLoginBtn = document.getElementById('tab-login-btn');
    const tabRegisterBtn = document.getElementById('tab-register-btn');
    const authError = document.getElementById('auth-error');
    const demoDoctorBtn = document.getElementById('demo-doctor-btn');
    const demoPatientBtn = document.getElementById('demo-patient-btn');

    // Doctor Notification Elements
    const doctorNotifBell = document.getElementById('doctor-notif-bell');
    const doctorNotifsSection = document.getElementById('doctor-notifications-section');
    const notifCountBadge = document.getElementById('notif-count');
    const unreadCountBadge = document.getElementById('unread-count-badge');
    const notificationsContainer = document.getElementById('notifications-container');
    const refreshNotifsBtn = document.getElementById('refresh-notifs-btn');

    // Audio Subtabs & Inputs
    const subtabRecord = document.getElementById('subtab-record');
    const subtabUpload = document.getElementById('subtab-upload');
    const modeRecordPanel = document.getElementById('mode-record-panel');
    const modeUploadPanel = document.getElementById('mode-upload-panel');

    const btnStartRecord = document.getElementById('btn-start-record');
    const btnStopRecord = document.getElementById('btn-stop-record');
    const visualizerBox = document.getElementById('visualizer-box');
    const recordTimer = document.getElementById('record-timer');
    const audioPreviewBox = document.getElementById('audio-preview-box');
    const recordedAudioElement = document.getElementById('recorded-audio-element');

    const dropzone = document.getElementById('dropzone');
    const audioFileInput = document.getElementById('audio-file-input');
    const selectedFileName = document.getElementById('selected-file-name');
    const btnUseSample = document.getElementById('btn-use-sample');
    const btnRunPipeline = document.getElementById('btn-run-pipeline');

    // Tremor Input Elements
    const subtabTremorSegment = document.getElementById('subtab-tremor-segment');
    const subtabTremorUpload = document.getElementById('subtab-tremor-upload');
    const modeTremorSegmentPanel = document.getElementById('mode-tremor-segment-panel');
    const modeTremorUploadPanel = document.getElementById('mode-tremor-upload-panel');
    const tremorSegmentInput = document.getElementById('tremor-segment-input');
    const tremorSegmentStatus = document.getElementById('tremor-segment-status');
    const tremorDropzone = document.getElementById('tremor-dropzone');
    const tremorFileInput = document.getElementById('tremor-file-input');
    const selectedTremorFileName = document.getElementById('selected-tremor-file-name');

    // Results Elements
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContent = document.getElementById('results-content');
    const highRiskBanner = document.getElementById('high-risk-banner');

    // Model 1: Voice
    const voiceClassBadge = document.getElementById('voice-class-badge');
    const voiceGaugeFill = document.getElementById('voice-gauge-fill');
    const voiceScoreVal = document.getElementById('voice-score-val');
    const rfScoreVal = document.getElementById('rf-score-val');
    const gbScoreVal = document.getElementById('gb-score-val');
    const audioFilenameVal = document.getElementById('audio-filename-val');

    // Model 2: Alcohol
    const alcoholRiskBadge = document.getElementById('alcohol-risk-badge');
    const alcoholGaugeFill = document.getElementById('alcohol-gauge-fill');
    const alcoholProbVal = document.getElementById('alcohol-prob-val');
    const resPatientAge = document.getElementById('res-patient-age');
    const resDrinksVal = document.getElementById('res-drinks-val');

    // Model 3: Tremor
    const tremorClassBadge = document.getElementById('tremor-class-badge');
    const tremorGaugeFill = document.getElementById('tremor-gauge-fill');
    const tremorScoreVal = document.getElementById('tremor-score-val');
    const tremorEtVal = document.getElementById('tremor-et-val');
    const tremorRfVal = document.getElementById('tremor-rf-val');
    const tremorGbVal = document.getElementById('tremor-gb-val');
    const tremorWindowsVal = document.getElementById('tremor-windows-val');
    const tremorSourceVal = document.getElementById('tremor-source-val');
    const tremorCvVal = document.getElementById('tremor-cv-val');

    // Tri-Modal Synthesis Elements
    const synthVoiceStatus = document.getElementById('synth-voice-status');
    const synthAlcoholStatus = document.getElementById('synth-alcohol-status');
    const synthTremorStatus = document.getElementById('synth-tremor-status');

    // History & Modal
    const historyTbody = document.getElementById('history-tbody');
    const btnRefreshHistory = document.getElementById('btn-refresh-history');
    const detailModal = document.getElementById('detail-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalBodyContent = document.getElementById('modal-body-content');
    const loadingSpinner = document.getElementById('loading-spinner');

    // =========================================================================
    // 1. INITIALIZATION & SESSION CHECK
    // =========================================================================

    async function initApp() {
        if (authToken) {
            try {
                const res = await fetch(`${API_BASE}/api/auth/me`, {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                if (res.ok) {
                    currentUser = await res.json();
                    setupDashboardForUser();
                    return;
                }
            } catch (err) {
                console.error("Session verification error:", err);
            }
        }
        showAuthScreen();
    }


    function showAuthScreen() {
        // Clear the current authenticated session.
        authToken = null;
        currentUser = null;
        localStorage.removeItem('parkinsons_jwt_token');

        // Stop any active recording when logging out.
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }

        clearInterval(recordTimerInterval);
        recordTimerInterval = null;

        // Close any open modal or loading overlay.
        const detailModal = document.getElementById('detail-modal');
        const loadingSpinner = document.getElementById('loading-spinner');

        if (detailModal) detailModal.style.display = 'none';
        if (loadingSpinner) loadingSpinner.style.display = 'none';

        // Explicitly hide the entire dashboard and its navigation controls.
        dashboardScreen.style.display = 'none';
        navUserControls.style.display = 'none';

        // Show the authentication screen.
        authScreen.style.display = 'flex';

        // Reset to the login tab.
        tabLoginBtn.classList.add('active');
        tabRegisterBtn.classList.remove('active');
        loginForm.classList.add('active');
        registerForm.classList.remove('active');
        hideAuthError();

        // Return the viewport to the top.
        window.scrollTo({ top: 0, behavior: 'instant' });
    }


    function setupDashboardForUser() {
        authScreen.style.display = 'none';
        dashboardScreen.style.display = 'block';
        navUserControls.style.display = 'flex';

        userDisplayName.textContent = currentUser.full_name;
        userRoleTag.textContent = currentUser.role.toUpperCase();

        if (currentUser.role === 'doctor') {
            doctorNotifBell.style.display = 'block';
            doctorNotifsSection.style.display = 'block';
            loadDoctorNotifications();
        } else {
            doctorNotifBell.style.display = 'none';
            doctorNotifsSection.style.display = 'none';
        }

        loadHistory();

        // Animate dashboard entrance
        dashboardScreen.style.opacity = '0';
        dashboardScreen.style.transform = 'translateY(20px)';
        setTimeout(() => {
            dashboardScreen.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            dashboardScreen.style.opacity = '1';
            dashboardScreen.style.transform = 'translateY(0)';
        }, 50);
    }

    // =========================================================================
    // 2. AUTHENTICATION HANDLERS
    // =========================================================================

    tabLoginBtn.addEventListener('click', () => {
        tabLoginBtn.classList.add('active');
        tabRegisterBtn.classList.remove('active');
        loginForm.classList.add('active');
        registerForm.classList.remove('active');
        hideAuthError();
    });

    tabRegisterBtn.addEventListener('click', () => {
        tabRegisterBtn.classList.add('active');
        tabLoginBtn.classList.remove('active');
        registerForm.classList.add('active');
        loginForm.classList.remove('active');
        hideAuthError();
    });

    function showAuthError(msg) {
        authError.textContent = msg;
        authError.style.display = 'block';
    }

    function hideAuthError() {
        authError.style.display = 'none';
    }

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideAuthError();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;

        try {
            const res = await fetch(`${API_BASE}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Login failed');

            authToken = data.access_token;
            currentUser = data.user;
            localStorage.setItem('parkinsons_jwt_token', authToken);
            setupDashboardForUser();
        } catch (err) {
            showAuthError(err.message);
        }
    });

    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideAuthError();
        const full_name = document.getElementById('reg-name').value;
        const email = document.getElementById('reg-email').value;
        const password = document.getElementById('reg-password').value;
        const role = document.getElementById('reg-role').value;

        try {
            const res = await fetch(`${API_BASE}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ full_name, email, password, role })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Registration failed');

            authToken = data.access_token;
            currentUser = data.user;
            localStorage.setItem('parkinsons_jwt_token', authToken);
            setupDashboardForUser();
        } catch (err) {
            showAuthError(err.message);
        }
    });

    demoDoctorBtn.addEventListener('click', () => {
        document.getElementById('login-email').value = 'doctor@hospital.org';
        document.getElementById('login-password').value = 'DocPass123';
        loginForm.requestSubmit();
    });

    demoPatientBtn.addEventListener('click', () => {
        document.getElementById('login-email').value = 'patient@example.com';
        document.getElementById('login-password').value = 'PatPass123';
        loginForm.requestSubmit();
    });

    logoutBtn.addEventListener('click', () => {
        showAuthScreen();
    });

    // =========================================================================
    // 3. AUDIO SUBTABS & LIVE WEB RECORDER
    // =========================================================================

    subtabRecord.addEventListener('click', () => {
        subtabRecord.classList.add('active');
        subtabUpload.classList.remove('active');
        modeRecordPanel.classList.add('active');
        modeUploadPanel.classList.remove('active');
    });

    subtabUpload.addEventListener('click', () => {
        subtabUpload.classList.add('active');
        subtabRecord.classList.remove('active');
        modeUploadPanel.classList.add('active');
        modeRecordPanel.classList.remove('active');
    });


    const btnDeleteRecording = document.getElementById('btn-delete-recording');

    btnStartRecord.addEventListener('click', async () => {
        try {
            // Clear any previous recording before starting again.
            recordedAudioBlob = null;
            audioChunks = [];

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true
            });

            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                if (audioChunks.length === 0) {
                    stream.getTracks().forEach(track => track.stop());
                    return;
                }

                recordedAudioBlob = new Blob(audioChunks, {
                    type: mediaRecorder.mimeType || 'audio/webm'
                });

                const audioUrl = URL.createObjectURL(recordedAudioBlob);
                recordedAudioElement.src = audioUrl;
                audioPreviewBox.style.display = 'block';

                if (btnDeleteRecording) {
                    btnDeleteRecording.style.display = 'inline-flex';
                }

                selectedUploadFile = null;
                useSampleAudioFlag = false;

                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();

            btnStartRecord.style.display = 'none';
            btnStopRecord.style.display = 'inline-flex';

            if (btnDeleteRecording) {
                btnDeleteRecording.style.display = 'none';
            }

            visualizerBox.classList.add('recording');

            recordSeconds = 0;
            recordTimer.textContent = '00:00';

            clearInterval(recordTimerInterval);

            recordTimerInterval = setInterval(() => {
                recordSeconds++;

                const mins = String(
                    Math.floor(recordSeconds / 60)
                ).padStart(2, '0');

                const secs = String(
                    recordSeconds % 60
                ).padStart(2, '0');

                recordTimer.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            alert(
                'Microphone permission denied or unsupported device. Please use File Upload instead.'
            );
            console.error('Microphone access error:', err);
        }
    });

    btnStopRecord.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }

        clearInterval(recordTimerInterval);
        recordTimerInterval = null;

        btnStartRecord.style.display = 'inline-flex';
        btnStopRecord.style.display = 'none';

        visualizerBox.classList.remove('recording');
    });

    if (btnDeleteRecording) {
        btnDeleteRecording.addEventListener('click', () => {
            // Stop playback and release the old audio URL.
            if (recordedAudioElement) {
                recordedAudioElement.pause();

                const oldUrl = recordedAudioElement.src;

                recordedAudioElement.removeAttribute('src');
                recordedAudioElement.load();

                if (oldUrl.startsWith('blob:')) {
                    URL.revokeObjectURL(oldUrl);
                }
            }

            // Clear the recording from application state.
            recordedAudioBlob = null;
            audioChunks = [];
            selectedUploadFile = null;
            useSampleAudioFlag = false;

            // Clear any uploaded file.
            if (audioFileInput) {
                audioFileInput.value = '';
            }

            if (selectedFileName) {
                selectedFileName.textContent = 'No file selected';
            }

            // Hide the preview and delete button.
            audioPreviewBox.style.display = 'none';
            btnDeleteRecording.style.display = 'none';

            // Reset the timer and recording controls.
            recordSeconds = 0;
            recordTimer.textContent = '00:00';

            btnStartRecord.style.display = 'inline-flex';
            btnStartRecord.disabled = false;
            btnStopRecord.style.display = 'none';

            visualizerBox.classList.remove('recording');
        });
    }



    btnStopRecord.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        clearInterval(recordTimerInterval);
        btnStartRecord.style.display = 'inline-flex';
        btnStopRecord.style.display = 'none';
        visualizerBox.classList.remove('recording');
    });

    // File Upload Handling
    dropzone.addEventListener('click', () => audioFileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary-color)';
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'var(--border-color)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--border-color)';
        if (e.dataTransfer.files.length) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelected(e.target.files[0]);
        }
    });

    function handleFileSelected(file) {
        selectedUploadFile = file;
        recordedAudioBlob = null;
        useSampleAudioFlag = false;
        selectedFileName.textContent = `Selected File: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        selectedFileName.style.color = 'var(--accent-color)';
    }

    btnUseSample.addEventListener('click', () => {
        useSampleAudioFlag = true;
        selectedUploadFile = null;
        recordedAudioBlob = null;
        selectedFileName.textContent = `Preset Selected: Pre-recorded Sample WAV (my_voice.wav)`;
        selectedFileName.style.color = 'var(--success-color)';
    });

    // =========================================================================
    // 3. TREMOR INPUT HANDLERS
    // =========================================================================

    if (subtabTremorSegment && subtabTremorUpload) {
        subtabTremorSegment.addEventListener('click', () => {
            subtabTremorSegment.classList.add('active');
            subtabTremorUpload.classList.remove('active');
            modeTremorSegmentPanel.style.display = 'block';
            modeTremorUploadPanel.style.display = 'none';
            useTremorSegmentMode = true;
        });

        subtabTremorUpload.addEventListener('click', () => {
            subtabTremorUpload.classList.add('active');
            subtabTremorSegment.classList.remove('active');
            modeTremorUploadPanel.style.display = 'block';
            modeTremorSegmentPanel.style.display = 'none';
            useTremorSegmentMode = false;
        });
    }

    if (tremorSegmentInput) {
        tremorSegmentInput.addEventListener('input', (e) => {
            const val = parseInt(e.target.value) || 1;
            const start = (val - 1) * 250 + 1;
            const end = val * 250;
            tremorSegmentStatus.textContent = `Segment #${val} (Rows ${start}–${end}, 9 analysis windows)`;
        });
    }

    document.querySelectorAll('.preset-segment-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const seg = btn.dataset.seg;
            if (tremorSegmentInput) {
                tremorSegmentInput.value = seg;
                const val = parseInt(seg);
                const start = (val - 1) * 250 + 1;
                const end = val * 250;
                tremorSegmentStatus.textContent = `Segment #${val} (Rows ${start}–${end}, 9 analysis windows)`;
                if (subtabTremorSegment) subtabTremorSegment.click();
            }
        });
    });

    if (tremorDropzone && tremorFileInput) {
        tremorDropzone.addEventListener('click', () => tremorFileInput.click());

        tremorDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            tremorDropzone.style.borderColor = 'var(--primary-color)';
        });

        tremorDropzone.addEventListener('dragleave', () => {
            tremorDropzone.style.borderColor = 'var(--border-color)';
        });

        tremorDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            tremorDropzone.style.borderColor = 'var(--border-color)';
            if (e.dataTransfer.files.length) {
                handleTremorFileSelected(e.dataTransfer.files[0]);
            }
        });

        tremorFileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleTremorFileSelected(e.target.files[0]);
            }
        });
    }

    function handleTremorFileSelected(file) {
        selectedTremorUploadFile = file;
        useTremorSegmentMode = false;
        selectedTremorFileName.textContent = `Selected CSV: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        selectedTremorFileName.style.color = 'var(--accent-color)';
    }

    // =========================================================================
    // 4. ML PIPELINE SUBMISSION
    // =========================================================================

    btnRunPipeline.addEventListener('click', async () => {
        const patient_name = document.getElementById('patient-name').value.trim();
        const patient_age = parseFloat(document.getElementById('patient-age').value);
        const patient_gender = document.getElementById('patient-gender').value;
        const avg_drinks = parseFloat(document.getElementById('avg-drinks').value);

        if (!patient_name) {
            alert("Please enter patient name / ID.");
            return;
        }
        if (isNaN(patient_age) || patient_age < 18 || patient_age > 120) {
            alert("Please enter a valid age between 18 and 120.");
            return;
        }
        if (isNaN(avg_drinks) || avg_drinks < 0 || avg_drinks > 25) {
            alert("Please enter valid alcohol drinks per day (0 - 25).");
            return;
        }

        const formData = new FormData();
        formData.append('patient_name', patient_name);
        formData.append('patient_age', patient_age);
        formData.append('patient_gender', patient_gender);
        formData.append('avg_drinks', avg_drinks);

        if (recordedAudioBlob) {
            formData.append('audio_file', recordedAudioBlob, 'live_recording.wav');
        } else if (selectedUploadFile) {
            formData.append('audio_file', selectedUploadFile);
        } else {
            formData.append('use_sample_audio', 'true');
        }

        // Tremor parameters
        if (!useTremorSegmentMode && selectedTremorUploadFile) {
            formData.append('tremor_file', selectedTremorUploadFile);
        } else {
            const segVal = tremorSegmentInput ? (parseInt(tremorSegmentInput.value) || 1) : 1;
            formData.append('tremor_segment', segVal);
            formData.append('use_sample_tremor', 'true');
        }

        loadingSpinner.style.display = 'flex';
        const spinnerText = document.getElementById('spinner-text');
        if (spinnerText) spinnerText.textContent = "Executing Voice, Alcohol, and Tremor ML Models...";

        try {
            const res = await fetch(`${API_BASE}/api/analysis/predict`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${authToken}` },
                body: formData
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Prediction failed');

            displayAnalysisResults(data);
            loadHistory();
            if (currentUser.role === 'doctor') {
                loadDoctorNotifications();
            }

        } catch (err) {
            alert(`Execution Error: ${err.message}`);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });

    function displayAnalysisResults(data) {
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'flex';

        // Animate results entrance
        resultsContent.style.opacity = '0';
        resultsContent.style.transform = 'translateY(20px)';
        setTimeout(() => {
            resultsContent.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            resultsContent.style.opacity = '1';
            resultsContent.style.transform = 'translateY(0)';
        }, 50);

        // High Risk Alert Banner
        if (data.alert_triggered) {
            highRiskBanner.style.display = 'flex';
        } else {
            highRiskBanner.style.display = 'none';
        }

        // Model 1: Voice Model Results
        const v = data.voice_analysis;
        const vScore = v.combined_score_pct.toFixed(1);
        voiceScoreVal.textContent = `${vScore}%`;
        voiceGaugeFill.style.width = `${vScore}%`;
        rfScoreVal.textContent = `${v.rf_score_pct.toFixed(1)}%`;
        gbScoreVal.textContent = `${v.gb_score_pct.toFixed(1)}%`;
        audioFilenameVal.textContent = v.audio_source;

        if (v.predicted_class === 1) {
            voiceClassBadge.textContent = "Parkinson's (Acoustic Pattern)";
            voiceClassBadge.className = 'result-class-badge badge-positive';
        } else {
            voiceClassBadge.textContent = "Healthy / Control";
            voiceClassBadge.className = 'result-class-badge badge-negative';
        }

        // Model 2: Alcohol Model Results
        const a = data.alcohol_analysis;
        const aProb = a.probability_pct.toFixed(1);
        alcoholProbVal.textContent = `${aProb}%`;
        alcoholGaugeFill.style.width = `${aProb}%`;
        resPatientAge.textContent = `${data.patient_age} yrs`;
        resDrinksVal.textContent = `${data.avg_drinks} drinks/day`;

        alcoholRiskBadge.textContent = `${a.risk_level} RISK`;
        if (a.risk_level === 'HIGH') {
            alcoholRiskBadge.className = 'result-risk-badge badge-risk-high';
        } else if (a.risk_level === 'MEDIUM') {
            alcoholRiskBadge.className = 'result-risk-badge badge-risk-med';
        } else {
            alcoholRiskBadge.className = 'result-risk-badge badge-risk-low';
        }

        // Model 3: Kinematic Tremor Model Results
        if (data.tremor_analysis && data.tremor_analysis.available) {
            const t = data.tremor_analysis;
            const tScore = t.combined_score_pct.toFixed(1);
            tremorScoreVal.textContent = `${tScore}%`;
            tremorGaugeFill.style.width = `${tScore}%`;
            tremorEtVal.textContent = `${t.et_score_pct.toFixed(1)}%`;
            tremorRfVal.textContent = `${t.rf_score_pct.toFixed(1)}%`;
            tremorGbVal.textContent = `${t.gb_score_pct.toFixed(1)}%`;
            tremorWindowsVal.textContent = `${t.windows_analysed} windows`;
            tremorSourceVal.textContent = t.source || `Segment #${t.dataset_number || 1}`;

            if (t.predicted_class === 1) {
                tremorClassBadge.textContent = "Parkinson's Tremor Pattern";
                tremorClassBadge.className = 'result-class-badge badge-positive';
            } else {
                tremorClassBadge.textContent = "Healthy / Control (No Tremor)";
                tremorClassBadge.className = 'result-class-badge badge-negative';
            }

            if (t.dataset_true_label && t.match !== null) {
                tremorCvVal.textContent = `Time-Blocked CV (${t.match ? 'Agreement: YES' : 'Agreement: NO'})`;
            } else {
                tremorCvVal.textContent = "Out-of-Fold Ensemble (Honest Test)";
            }

            // Tri-Modal Synthesis Display
            if (synthVoiceStatus) {
                synthVoiceStatus.textContent = v.predicted_class === 1 ? "⚠️ Dysphonic Articulation" : "✅ Normal Phonation";
                synthVoiceStatus.style.color = v.predicted_class === 1 ? "var(--danger-color)" : "var(--success-color)";
            }
            if (synthAlcoholStatus) {
                synthAlcoholStatus.textContent = `${a.risk_level} Risk (${a.probability_pct.toFixed(0)}%)`;
                synthAlcoholStatus.style.color = a.risk_level === 'HIGH' ? "var(--danger-color)" : (a.risk_level === 'MEDIUM' ? "var(--warning-color)" : "var(--success-color)");
            }
            if (synthTremorStatus) {
                synthTremorStatus.textContent = t.predicted_class === 1 ? "⚠️ Tremor Biomarker (+)" : "✅ Normal Biomechanics (-)";
                synthTremorStatus.style.color = t.predicted_class === 1 ? "var(--danger-color)" : "var(--success-color)";
            }
        }
    }

    // =========================================================================
    // 5. DOCTOR NOTIFICATIONS & HISTORY
    // =========================================================================

    async function loadDoctorNotifications() {
        try {
            const res = await fetch(`${API_BASE}/api/notifications`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;

            const data = await res.json();
            notifCountBadge.textContent = data.unread_count;
            unreadCountBadge.textContent = `${data.unread_count} New`;

            if (data.notifications.length === 0) {
                notificationsContainer.innerHTML = `<div class="empty-state">No clinical risk alerts generated yet.</div>`;
                return;
            }

            notificationsContainer.innerHTML = data.notifications.map(n => `
                <div class="notif-card ${n.is_read ? 'read' : ''}">
                    <div>
                        <div class="notif-title">🔔 HIGH PRIORITY: ${n.patient_name} (Age: ${n.patient_age})</div>
                        <div class="notif-meta">
                            Voice Confidence: <strong>${n.voice_confidence.toFixed(1)}%</strong> | Generated: ${n.created_at}
                        </div>
                    </div>
                    <div style="display:flex; gap:0.5rem;">
                        ${!n.is_read ? `<button class="btn btn-secondary btn-xs mark-read-btn" data-id="${n.id}">Mark Read</button>` : ''}
                        <button class="btn btn-primary btn-xs view-detail-btn" data-id="${n.analysis_id}">View Analysis</button>
                    </div>
                </div>
            `).join('');

            // Attach notification action handlers
            document.querySelectorAll('.mark-read-btn').forEach(b => {
                b.addEventListener('click', async (e) => {
                    const id = e.target.dataset.id;
                    await fetch(`${API_BASE}/api/notifications/${id}/read`, {
                        method: 'PATCH',
                        headers: { 'Authorization': `Bearer ${authToken}` }
                    });
                    loadDoctorNotifications();
                });
            });

            document.querySelectorAll('.view-detail-btn').forEach(b => {
                b.addEventListener('click', (e) => {
                    openAnalysisDetailModal(e.target.dataset.id);
                });
            });

        } catch (err) {
            console.error("Failed to load doctor notifications:", err);
        }
    }

    refreshNotifsBtn.addEventListener('click', loadDoctorNotifications);

    async function loadHistory() {
        try {
            const res = await fetch(`${API_BASE}/api/analysis/history`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;

            const records = await res.json();

            if (records.length === 0) {
                historyTbody.innerHTML = `<tr><td colspan="9" class="text-center">No analysis records found.</td></tr>`;
                return;
            }

            historyTbody.innerHTML = records.map(r => `
                <tr>
                    <td>${r.created_at}</td>
                    <td><strong>${r.patient_name}</strong> (${r.patient_age}y)</td>
                    <td>${r.voice_label.includes("Parkinson") ? '<span class="badge badge-positive">Parkinson\'s</span>' : '<span class="badge badge-negative">Healthy</span>'}</td>
                    <td><strong>${r.voice_score.toFixed(1)}%</strong></td>
                    <td><span class="badge ${r.alcohol_risk === 'HIGH' ? 'badge-risk-high' : (r.alcohol_risk === 'MEDIUM' ? 'badge-risk-med' : 'badge-risk-low')}">${r.alcohol_risk} (${r.alcohol_prob.toFixed(1)}%)</span></td>
                    <td>${r.tremor_label ? (r.tremor_label.includes("Parkinson") ? '<span class="badge badge-positive">Tremor Pattern</span>' : '<span class="badge badge-negative">Healthy</span>') : '<span class="badge badge-grey">N/A</span>'}</td>
                    <td><strong>${r.tremor_score !== null && r.tremor_score !== undefined ? `${r.tremor_score.toFixed(1)}%` : '—'}</strong></td>
                    <td>${r.alert_triggered ? '🔔 HIGH' : '—'}</td>
                    <td><button class="btn btn-secondary btn-xs history-detail-btn" data-id="${r.id}">Details</button></td>
                </tr>
            `).join('');

            document.querySelectorAll('.history-detail-btn').forEach(b => {
                b.addEventListener('click', (e) => {
                    openAnalysisDetailModal(e.target.dataset.id);
                });
            });

        } catch (err) {
            console.error("Failed to load analysis history:", err);
        }
    }

    btnRefreshHistory.addEventListener('click', loadHistory);

    async function openAnalysisDetailModal(analysisId) {
        try {
            const res = await fetch(`${API_BASE}/api/analysis/history/${analysisId}`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) throw new Error("Could not fetch detail");

            const r = await res.json();

            modalBodyContent.innerHTML = `
                <div class="sub-metrics-grid" style="font-size:0.9rem; gap:0.75rem;">
                    <div><strong>Patient Name:</strong> ${r.patient_name}</div>
                    <div><strong>Age / Gender:</strong> ${r.patient_age} yrs | ${r.patient_gender}</div>
                    <div><strong>Avg Alcohol Input:</strong> ${r.avg_drinks} drinks/day</div>
                    <div><strong>Date & Time:</strong> ${r.created_at}</div>
                    <div><strong>Voice Ensemble Score:</strong> ${r.voice_score.toFixed(2)}%</div>
                    <div><strong>Voice Prediction:</strong> ${r.voice_label}</div>
                    <div><strong>Random Forest Score:</strong> ${r.rf_score.toFixed(2)}%</div>
                    <div><strong>Gradient Boosting Score:</strong> ${r.gb_score.toFixed(2)}%</div>
                    <div><strong>Alcohol Model Probability:</strong> ${r.alcohol_prob.toFixed(2)}% (${r.alcohol_risk} RISK)</div>
                    <div><strong>Tremor Model Score:</strong> ${r.tremor_score !== null && r.tremor_score !== undefined ? `${r.tremor_score.toFixed(2)}%` : 'N/A'}</div>
                    <div><strong>Tremor Prediction:</strong> ${r.tremor_label || 'N/A'}</div>
                    <div><strong>Tremor Source:</strong> ${r.tremor_source || 'N/A'}</div>
                    <div><strong>Clinical Alert Triggered:</strong> ${r.alert_triggered ? 'Yes (🔔 Doctor Notified)' : 'No'}</div>
                    <div><strong>Audio Source File:</strong> ${r.audio_filename}</div>
                </div>
            `;
            detailModal.style.display = 'flex';

        } catch (err) {
            alert(err.message);
        }
    }

    modalCloseBtn.addEventListener('click', () => {
        detailModal.style.display = 'none';
    });

    detailModal.addEventListener('click', (e) => {
        if (e.target === detailModal) {
            detailModal.style.display = 'none';
        }
    });

    // Start App
    initApp();
});
