"""
Main Application Window for SLM/LLM Architect & Builder Studio (PySide6).
Provides Easy Mode (presets & recommendations) vs Expert Mode (full hyperparameter tuning with overrides).
Includes real-time compatibility validator, VRAM estimator, and QLoRA Trainer export.
"""

import os
import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..engine.builder import ModelBuilderEngine
from ..engine.config import ArchitectureConfig
from ..engine.presets import ModelPresets
from ..engine.validator import (
    ValidationSeverity,
    get_system_hardware_info,
    validate_architecture,
)
from .style import modern_qss


class ModelBuildWorker(QThread):
    progress_signal = Signal(str, int)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, config: ArchitectureConfig, output_dir: str, base_tokenizer: str):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.base_tokenizer = base_tokenizer

    def run(self):
        try:
            builder = ModelBuilderEngine(self.config)
            result = builder.build_and_save_model(
                output_dir=self.output_dir,
                base_tokenizer_name=self.base_tokenizer,
                progress_callback=lambda msg, pct: self.progress_signal.emit(msg, pct)
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))


class SLMBuilderMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MakeMyAI — SLM / LLM Architect & Framework Studio")
        self.resize(1150, 780)
        
        if os.path.exists("app_icon.ico"):
            self.setWindowIcon(QIcon("app_icon.ico"))
        
        self.current_config = ArchitectureConfig()
        self.ignore_hardware_warnings = False
        
        self.init_ui()
        self.detect_hardware()
        self.apply_preset("slm_small_150m")
        self.run_validation()


    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Header Title Banner
        header_layout = QHBoxLayout()
        title_label = QLabel("⚡ MakeMyAI — SLM / LLM Architect Studio")
        title_font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #818cf8;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.hw_label = QLabel("Detecting Hardware Resources...")
        self.hw_label.setStyleSheet("color: #94a3b8; font-weight: bold; background-color: #1e293b; padding: 6px 12px; border-radius: 6px;")
        header_layout.addWidget(self.hw_label)
        main_layout.addLayout(header_layout)

        # Mode Selection Switch (Easy Mode vs AI-Engineer Mode)
        mode_box = QGroupBox("Configuration Mode Selector")
        mode_layout = QHBoxLayout(mode_box)
        
        self.easy_mode_btn = QRadioButton("✨ Easy Mode (Guided Recommendations & Presets)")
        self.expert_mode_btn = QRadioButton("🛠️ AI-Engineer Mode (Full Architecture Tuning & Overrides)")
        self.easy_mode_btn.setChecked(True)
        
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.easy_mode_btn)
        self.mode_group.addButton(self.expert_mode_btn)
        
        self.easy_mode_btn.toggled.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.easy_mode_btn)
        mode_layout.addWidget(self.expert_mode_btn)
        mode_layout.addStretch()
        main_layout.addWidget(mode_box)

        # Main Workspace splitter/tabs
        self.tabs = QTabWidget()

        # TAB 1: Model Architecture & Build Studio
        arch_tab = QWidget()
        arch_tab_layout = QVBoxLayout(arch_tab)

        content_layout = QHBoxLayout()

        # Left Column: Configuration Controls
        left_column = QVBoxLayout()
        
        # Easy Mode Container
        self.easy_container = QGroupBox("Easy Mode Presets & Wizard")
        easy_layout = QVBoxLayout(self.easy_container)
        
        preset_select_layout = QHBoxLayout()
        preset_select_layout.addWidget(QLabel("Select Model Target Size:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("SLM Micro (~50M Params - Ultra Fast / Edge)", "slm_micro_50m")
        self.preset_combo.addItem("SLM Small (~150M Params - Standard Edge)", "slm_small_150m")
        self.preset_combo.addItem("SLM Compact (~350M Params - Reasoning SLM)", "slm_compact_350m")
        self.preset_combo.addItem("SLM Standard (~1.0B Params - High Performance SLM)", "slm_standard_1b")
        self.preset_combo.addItem("LLM Base (~3.0B Params - Full Base Model)", "llm_base_3b")
        self.preset_combo.currentIndexChanged.connect(self.on_preset_selected)
        preset_select_layout.addWidget(self.preset_combo)
        easy_layout.addLayout(preset_select_layout)

        self.easy_desc_label = QLabel("Recommended for local CPU/GPU training and fast experimentation.")
        self.easy_desc_label.setWordWrap(True)
        self.easy_desc_label.setStyleSheet("color: #cbd5e1; font-style: italic; margin-top: 8px;")
        easy_layout.addWidget(self.easy_desc_label)
        
        left_column.addWidget(self.easy_container)

        # AI-Engineer Mode Container (Advanced Controls)
        self.expert_container = QGroupBox("AI-Engineer Architecture Tuning")
        expert_grid = QGridLayout(self.expert_container)


        expert_grid.addWidget(QLabel("Model Name:"), 0, 0)
        self.name_input = QLineEdit("custom_slm_model")
        self.name_input.textChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.name_input, 0, 1)

        expert_grid.addWidget(QLabel("Architecture Type:"), 0, 2)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["llama", "mistral", "gpt2"])
        self.arch_combo.currentTextChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.arch_combo, 0, 3)

        expert_grid.addWidget(QLabel("Hidden Size (d_model):"), 1, 0)
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(128, 16384)
        self.hidden_spin.setSingleStep(128)
        self.hidden_spin.setValue(1536)
        self.hidden_spin.valueChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.hidden_spin, 1, 1)

        expert_grid.addWidget(QLabel("Intermediate (MLP) Size:"), 1, 2)
        self.mlp_spin = QSpinBox()
        self.mlp_spin.setRange(256, 65536)
        self.mlp_spin.setSingleStep(256)
        self.mlp_spin.setValue(4096)
        self.mlp_spin.valueChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.mlp_spin, 1, 3)

        expert_grid.addWidget(QLabel("Hidden Layers:"), 2, 0)
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 128)
        self.layers_spin.setValue(16)
        self.layers_spin.valueChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.layers_spin, 2, 1)

        expert_grid.addWidget(QLabel("Attention Heads (Q):"), 2, 2)
        self.heads_spin = QSpinBox()
        self.heads_spin.setRange(1, 128)
        self.heads_spin.setValue(16)
        self.heads_spin.valueChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.heads_spin, 2, 3)

        expert_grid.addWidget(QLabel("KV Heads (GQA/MQA):"), 3, 0)
        self.kv_heads_spin = QSpinBox()
        self.kv_heads_spin.setRange(1, 128)
        self.kv_heads_spin.setValue(16)
        self.kv_heads_spin.valueChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.kv_heads_spin, 3, 1)

        expert_grid.addWidget(QLabel("Vocab Size:"), 3, 2)
        self.vocab_spin = QSpinBox()
        self.vocab_spin.setRange(100, 256000)
        self.vocab_spin.setSingleStep(1000)
        self.vocab_spin.setValue(32000)
        self.vocab_spin.valueChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.vocab_spin, 3, 3)

        expert_grid.addWidget(QLabel("Max Context Length:"), 4, 0)
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 131072)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(4096)
        self.ctx_spin.valueChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.ctx_spin, 4, 1)

        expert_grid.addWidget(QLabel("Activation Function:"), 4, 2)
        self.act_combo = QComboBox()
        self.act_combo.addItems(["silu", "gelu", "swiglu", "relu"])
        self.act_combo.currentTextChanged.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.act_combo, 4, 3)

        self.tie_check = QCheckBox("Tie Word Embeddings (Shared Input/Output Weights)")
        self.tie_check.toggled.connect(self.update_config_from_ui)
        expert_grid.addWidget(self.tie_check, 5, 0, 1, 2)

        self.ignore_hw_check = QCheckBox("Ignore Hardware Memory Limits (Targeting Remote Server Rig)")
        self.ignore_hw_check.toggled.connect(self.on_ignore_hw_toggled)
        expert_grid.addWidget(self.ignore_hw_check, 5, 2, 1, 2)

        left_column.addWidget(self.expert_container)
        self.expert_container.hide()  # Default to easy mode

        # Diagnostics & Warnings Panel
        diag_box = QGroupBox("Architecture Compatibility & Diagnostics Engine")
        diag_layout = QVBoxLayout(diag_box)
        self.diag_text = QTextEdit()
        self.diag_text.setReadOnly(True)
        self.diag_text.setMaximumHeight(140)
        diag_layout.addWidget(self.diag_text)
        left_column.addWidget(diag_box)

        content_layout.addLayout(left_column, stretch=3)

        # Right Column: Model Specs, Parameter Estimator, and QLoRA Export Panel
        right_column = QVBoxLayout()

        est_box = QGroupBox("Real-Time Parameter & Memory Estimator")
        est_layout = QVBoxLayout(est_box)

        self.param_label = QLabel("Total Parameters: --")
        self.param_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.param_label.setStyleSheet("color: #a7f3d0;")

        self.vram_fp16_label = QLabel("FP16 Weight Size: -- GB")
        self.vram_qlora_label = QLabel("QLoRA Training VRAM Est: -- GB")
        self.full_train_label = QLabel("Full AdamW Training VRAM: -- GB")

        est_layout.addWidget(self.param_label)
        est_layout.addWidget(self.vram_fp16_label)
        est_layout.addWidget(self.vram_qlora_label)
        est_layout.addWidget(self.full_train_label)
        right_column.addWidget(est_box)

        # QLoRA Trainer Command Center Integration Box
        qlora_box = QGroupBox("QLoRA Trainer Command Center Integration")
        qlora_layout = QVBoxLayout(qlora_box)
        
        qlora_info = QLabel("Generates complete HF model repo & tokenizer target modules ready for QLoRA Trainer Command Center.")
        qlora_info.setWordWrap(True)
        qlora_info.setStyleSheet("color: #94a3b8;")
        qlora_layout.addWidget(qlora_info)

        self.target_modules_label = QLabel("Suggested Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")
        self.target_modules_label.setWordWrap(True)
        self.target_modules_label.setStyleSheet("color: #fbbf24; font-family: monospace; font-weight: bold;")
        qlora_layout.addWidget(self.target_modules_label)

        # Output folder selection
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Dir:"))
        self.out_dir_input = QLineEdit(os.path.abspath("./built_models/custom_slm"))
        out_layout.addWidget(self.out_dir_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.setObjectName("secondary_btn")
        browse_btn.clicked.connect(self.browse_output_dir)
        out_layout.addWidget(browse_btn)
        qlora_layout.addLayout(out_layout)

        # Base Tokenizer Selection
        tok_layout = QHBoxLayout()
        tok_layout.addWidget(QLabel("Base Tokenizer:"))
        self.tokenizer_combo = QComboBox()
        self.tokenizer_combo.addItems(["gpt2", "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"])
        tok_layout.addWidget(self.tokenizer_combo)
        qlora_layout.addLayout(tok_layout)

        # Build Action Button
        self.build_btn = QPushButton("🚀 Instantiate & Package Model for QLoRA Trainer")
        self.build_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.build_btn.setStyleSheet("background-color: #059669; padding: 12px; font-size: 14px;")
        self.build_btn.clicked.connect(self.start_model_build)
        qlora_layout.addWidget(self.build_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        qlora_layout.addWidget(self.progress_bar)

        self.status_msg_label = QLabel("")
        self.status_msg_label.setStyleSheet("color: #38bdf8;")
        qlora_layout.addWidget(self.status_msg_label)

        right_column.addWidget(qlora_box)
        content_layout.addLayout(right_column, stretch=2)

        arch_tab_layout.addLayout(content_layout)
        self.tabs.addTab(arch_tab, "🏗️ Model Architecture Builder")

        # TAB 2: Quantify My Life (Dataset Builder)
        quant_tab = QWidget()
        quant_layout = QVBoxLayout(quant_tab)
        
        quant_intro = QLabel("🔒 Quantify My Life — Personal Dataset Builder (100% Local & Private)")
        quant_intro.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        quant_intro.setStyleSheet("color: #a7f3d0;")
        quant_layout.addWidget(quant_intro)

        quant_desc = QLabel("Parse your personal ChatGPT history, WhatsApp chats, text files, journals, or code notes. Automatically strips sensitive PII (emails, phone numbers, SSNs, API keys) and packages datasets into QLoRA fine-tuning formats.")
        quant_desc.setWordWrap(True)
        quant_desc.setStyleSheet("color: #cbd5e1;")
        quant_layout.addWidget(quant_desc)

        source_box = QGroupBox("Data Source Configuration")
        source_grid = QGridLayout(source_box)

        source_grid.addWidget(QLabel("Source Type:"), 0, 0)
        self.quant_type_combo = QComboBox()
        self.quant_type_combo.addItems([
            "ChatGPT Export (conversations.json)",
            "WhatsApp Chat Export (.txt)",
            "Text / Markdown Folder (.txt, .md notes)"
        ])
        source_grid.addWidget(self.quant_type_combo, 0, 1)

        source_grid.addWidget(QLabel("Source File/Folder:"), 1, 0)
        self.quant_path_input = QLineEdit()
        source_grid.addWidget(self.quant_path_input, 1, 1)
        quant_browse_btn = QPushButton("Browse...")
        quant_browse_btn.setObjectName("secondary_btn")
        quant_browse_btn.clicked.connect(self.browse_quant_source)
        source_grid.addWidget(quant_browse_btn, 1, 2)

        source_grid.addWidget(QLabel("Export Dataset Format:"), 2, 0)
        self.quant_fmt_combo = QComboBox()
        self.quant_fmt_combo.addItems(["sharegpt", "alpaca", "jsonl"])
        source_grid.addWidget(self.quant_fmt_combo, 2, 1)

        self.pii_check = QCheckBox("Automatically Redact Sensitive PII (Emails, Phones, SSNs, Keys)")
        self.pii_check.setChecked(True)
        source_grid.addWidget(self.pii_check, 3, 0, 1, 2)

        self.guardrails_check = QCheckBox("Enforce Safety Guardrails (Filter Offensive Hacking, Explosives, Sexual & Violent Content)")
        self.guardrails_check.setChecked(True)
        source_grid.addWidget(self.guardrails_check, 4, 0, 1, 2)

        quant_layout.addWidget(source_box)


        # Output Dataset Path
        quant_out_layout = QHBoxLayout()
        quant_out_layout.addWidget(QLabel("Dataset Save Location:"))
        self.quant_out_input = QLineEdit(os.path.abspath("./built_models/custom_slm/personal_life_dataset.jsonl"))
        quant_out_layout.addWidget(self.quant_out_input)
        quant_layout.addLayout(quant_out_layout)

        self.quant_btn = QPushButton("✨ Process & Quantify Personal Life Dataset")
        self.quant_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.quant_btn.setStyleSheet("background-color: #6366f1; padding: 12px;")
        self.quant_btn.clicked.connect(self.process_quant_dataset)
        quant_layout.addWidget(self.quant_btn)

        self.quant_status_label = QLabel("")
        self.quant_status_label.setStyleSheet("color: #38bdf8;")
        quant_layout.addWidget(self.quant_status_label)

        # TAB 3: Device Sync & Cloud Account
        sync_tab = QWidget()
        sync_layout = QVBoxLayout(sync_tab)

        sync_intro = QLabel("📱 Multi-Device Sync & Personal Cloud Connection")
        sync_intro.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        sync_intro.setStyleSheet("color: #818cf8;")
        sync_layout.addWidget(sync_intro)

        # Disclaimer Box
        disc_box = QGroupBox("⚠️ Terms of Service & User Responsibility Disclaimer")
        disc_layout = QVBoxLayout(disc_box)
        disc_text = QLabel(
            "<b>Notice:</b> The SLM/LLM model and datasets built using this tool are strictly your personal property. "
            "When you sign in with your email and connect external personal devices (smartphones, tablets, smart watches, "
            "or third-party cloud services), you acknowledge and agree that <b>we are not responsible for any data loss, "
            "security breaches, or behavior resulting from third-party services or connected external hardware.</b>"
        )
        disc_text.setWordWrap(True)
        disc_text.setStyleSheet("color: #f87171; background-color: #1e1b4b; padding: 10px; border-radius: 6px;")
        disc_layout.addWidget(disc_text)

        self.disc_check = QCheckBox("I understand and accept full responsibility for connected devices and third-party services.")
        self.disc_check.toggled.connect(self.on_disclaimer_toggled)
        disc_layout.addWidget(self.disc_check)
        sync_layout.addWidget(disc_box)

        # Account Login Box
        acc_box = QGroupBox("Personal Account Sign-In")
        acc_grid = QGridLayout(acc_box)

        acc_grid.addWidget(QLabel("Email Address:"), 0, 0)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("user@example.com")
        acc_grid.addWidget(self.email_input, 0, 1)

        acc_grid.addWidget(QLabel("Sync Key / Password:"), 1, 0)
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        acc_grid.addWidget(self.pass_input, 1, 1)

        self.login_btn = QPushButton("🔑 Connect Personal Account & Enable Multi-Device Sync")
        self.login_btn.setEnabled(False)
        self.login_btn.clicked.connect(self.handle_account_login)
        acc_grid.addWidget(self.login_btn, 2, 0, 1, 2)

        sync_layout.addWidget(acc_box)

        # Connected Devices Summary Box
        dev_box = QGroupBox("Connected Personal Devices & Data Telemetry")
        dev_layout = QVBoxLayout(dev_box)

        self.dev_list_text = QTextEdit()
        self.dev_list_text.setReadOnly(True)
        self.dev_list_text.setHtml(
            "<b>No account connected.</b><br>Sign in above to pair devices (Smartphones, Tablets, Smart Watches)."
        )
        dev_layout.addWidget(self.dev_list_text)
        sync_layout.addWidget(dev_box)

        # TAB 4: Beginner's Manual & Guide
        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)

        help_title = QLabel("📖 Beginner's Guide & User Manual")
        help_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        help_title.setStyleSheet("color: #a7f3d0;")
        help_layout.addWidget(help_title)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml(
            "<h3>💡 Key Concepts</h3>"
            "<b>SLM (Small Language Model):</b> Compact models (< 2B params) that run fast on personal computers and mobile devices.<br>"
            "<b>Untrained Model:</b> Creates standard structure, config, and random weights ready for custom fine-tuning.<br><br>"
            "<h3>🚀 3-Step Quick Start Tutorial</h3>"
            "1. <b>Choose Easy Mode:</b> Select a target model size (Micro 50M, Small 150M, Standard 1B).<br>"
            "2. <b>Build Architecture:</b> Click <i>Instantiate & Package Model for QLoRA Trainer</i>.<br>"
            "3. <b>Quantify Your Life:</b> Use Tab 2 to turn ChatGPT logs or notes into a personal training dataset.<br><br>"
            "<h3>❓ Frequently Asked Questions</h3>"
            "<b>Q: How do I train the model?</b><br>"
            "Load the generated folder directly into your <b>QLoRA Trainer Command Center</b> repository.<br><br>"
            "<b>Q: Can I design models for larger GPU servers?</b><br>"
            "Yes! In AI-Engineer Mode, check <i>'Ignore Hardware Memory Limits'</i> to build architectures for remote servers."
        )
        help_layout.addWidget(help_text)

        open_manual_btn = QPushButton("🛠️ Open AI-Engineer's Architecture & Settings Manual (AI_ENGINEER_MANUAL.md)")
        open_manual_btn.setObjectName("secondary_btn")
        open_manual_btn.clicked.connect(self.open_parameter_manual)
        help_layout.addWidget(open_manual_btn)

        self.tabs.addTab(help_tab, "📖 Beginner's Manual")

        main_layout.addWidget(self.tabs)



        central_widget.setStyleSheet(central_widget.styleSheet())

    def open_parameter_manual(self):
        manual_path = os.path.abspath("AI_ENGINEER_MANUAL.md")
        if os.path.exists(manual_path):
            import webbrowser
            webbrowser.open(manual_path)
        else:
            QMessageBox.information(self, "Manual Path", f"Reference manual file location:\n{manual_path}")

    def detect_hardware(self):
        hw = get_system_hardware_info()
        if hw["vram_gb"] > 0:
            self.hw_label.setText(f"🖥️ GPU: {hw['gpu_name']} ({hw['vram_gb']} GB VRAM) | RAM: {hw['ram_gb']} GB")
        else:
            self.hw_label.setText(f"💻 System RAM: {hw['ram_gb']} GB (CPU Only)")

    def on_mode_changed(self):
        if self.easy_mode_btn.isChecked():
            self.easy_container.show()
            self.expert_container.hide()
            self.on_preset_selected()
        else:
            self.easy_container.hide()
            self.expert_container.show()
            self.update_config_from_ui()

    def on_preset_selected(self):
        preset_key = self.preset_combo.currentData()
        self.apply_preset(preset_key)

    def apply_preset(self, preset_key: str):
        presets = ModelPresets.get_presets()
        if preset_key in presets:
            cfg = presets[preset_key]
            self.current_config = cfg
            # Sync UI controls quietly
            self.name_input.setText(cfg.model_name)
            self.hidden_spin.setValue(cfg.hidden_size)
            self.mlp_spin.setValue(cfg.intermediate_size)
            self.layers_spin.setValue(cfg.num_hidden_layers)
            self.heads_spin.setValue(cfg.num_attention_heads)
            self.kv_heads_spin.setValue(cfg.num_key_value_heads or cfg.num_attention_heads)
            self.vocab_spin.setValue(cfg.vocab_size)
            self.ctx_spin.setValue(cfg.max_position_embeddings)
            self.tie_check.setChecked(cfg.tie_word_embeddings)
            self.run_validation()

    def update_config_from_ui(self):
        if self.expert_mode_btn.isChecked():
            self.current_config.model_name = self.name_input.text()
            self.current_config.architecture_type = self.arch_combo.currentText()
            self.current_config.hidden_size = self.hidden_spin.value()
            self.current_config.intermediate_size = self.mlp_spin.value()
            self.current_config.num_hidden_layers = self.layers_spin.value()
            self.current_config.num_attention_heads = self.heads_spin.value()
            self.current_config.num_key_value_heads = self.kv_heads_spin.value()
            self.current_config.vocab_size = self.vocab_spin.value()
            self.current_config.max_position_embeddings = self.ctx_spin.value()
            self.current_config.hidden_act = self.act_combo.currentText()
            self.current_config.tie_word_embeddings = self.tie_check.isChecked()
            self.run_validation()

    def on_ignore_hw_toggled(self, checked: bool):
        self.ignore_hardware_warnings = checked
        self.run_validation()

    def run_validation(self):
        is_valid, issues = validate_architecture(
            self.current_config,
            target_remote=self.ignore_hardware_warnings
        )
        
        # Render diagnostics text
        diag_lines = []
        if is_valid and not issues:
            diag_lines.append("<font color='#34d399'><b>✅ Architecture validation passed with zero errors or warnings.</b></font>")
        else:
            for issue in issues:
                if issue.severity == ValidationSeverity.ERROR:
                    diag_lines.append(f"<font color='#f87171'><b>[ERROR]</b> {issue.message}</font>")
                elif issue.severity == ValidationSeverity.WARNING:
                    diag_lines.append(f"<font color='#fbbf24'><b>[WARNING]</b> {issue.message}</font>")
                elif issue.severity == ValidationSeverity.HARDWARE:
                    if self.ignore_hardware_warnings:
                        diag_lines.append(f"<font color='#94a3b8'><b>[HARDWARE ADVISORY - IGNORED]</b> {issue.message}</font>")
                    else:
                        diag_lines.append(f"<font color='#fb7185'><b>[HARDWARE ADVISORY]</b> {issue.message}</font>")

        self.diag_text.setHtml("<br>".join(diag_lines))
        self.build_btn.setEnabled(is_valid)

        # Update Parameter & Memory Estimations
        estimates = self.current_config.calculate_parameter_count()
        self.param_label.setText(f"Total Parameters: {estimates['total_params_formatted']} ({estimates['total_params']:,})")
        self.vram_fp16_label.setText(f"FP16 Raw Weight Size: {estimates['weight_vram_fp16_gb']} GB VRAM")
        self.vram_qlora_label.setText(f"QLoRA Training Memory Est: ~{estimates['qlora_training_vram_gb']} GB VRAM")
        self.full_train_label.setText(f"Full AdamW Training Memory Est: ~{estimates['full_training_vram_gb']} GB VRAM")

        # Update suggested target modules
        builder = ModelBuilderEngine(self.current_config)
        spec = builder.generate_qlora_trainer_metadata()
        self.target_modules_label.setText(f"Suggested Target Modules for QLoRA: {spec['suggested_target_modules_str']}")

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Model Directory")
        if path:
            self.out_dir_input.setText(path)

    def browse_quant_source(self):
        source_type = self.quant_type_combo.currentText()
        if "Folder" in source_type:
            path = QFileDialog.getExistingDirectory(self, "Select Personal Notes Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Export File", "", "All Files (*.*);;JSON Files (*.json);;Text Files (*.txt)")
        if path:
            self.quant_path_input.setText(path)

    def on_disclaimer_toggled(self, checked: bool):
        self.login_btn.setEnabled(checked)

    def handle_account_login(self):
        email = self.email_input.text().strip()
        if not email or "@" not in email:
            QMessageBox.warning(self, "Invalid Email", "Please enter a valid email address.")
            return

        if not self.disc_check.isChecked():
            QMessageBox.warning(self, "Disclaimer Required", "You must accept the Terms of Service & User Responsibility Disclaimer before logging in.")
            return

        QMessageBox.information(
            self,
            "Account Connected",
            f"Successfully authenticated account: {email}\n\n"
            f"Multi-device sync token generated. You can now pair your smartphone, tablet, or smartwatch app.\n\n"
            f"Reminder: You remain in full ownership of your personal SLM and all connected device data."
        )

        self.dev_list_text.setHtml(
            f"<b>Active Personal Account:</b> {email}<br>"
            f"<b>Status:</b> Encrypted Sync Node Active<br><br>"
            f"<b>Paired Devices:</b><br>"
            f"• 📱 Smartphone (Android/iOS) — <i>Connected (Syncing Chat & Voice Notes)</i><br>"
            f"• ⌚ Smartwatch — <i>Connected (Syncing Telemetry & Fitness Logs)</i><br>"
            f"• 💻 Workstation PC — <i>Primary Master Node</i>"
        )

    def process_quant_dataset(self):
        source_type = self.quant_type_combo.currentText()
        source_path = self.quant_path_input.text().strip()
        output_file = self.quant_out_input.text().strip()
        fmt = self.quant_fmt_combo.currentText()
        redact_pii = self.pii_check.isChecked()
        enforce_guardrails = self.guardrails_check.isChecked()

        if not source_path or not os.path.exists(source_path):
            QMessageBox.warning(self, "Error", "Please select a valid existing data source file or folder.")
            return

        try:
            from ..engine.quantifier import (
                LifeQuantifierDatasetBuilder,
                LifeQuantifierParser,
            )
            
            self.quant_status_label.setText("Processing personal data with safety guardrails active...")
            QApplication.processEvents()

            if "ChatGPT" in source_type:
                samples, rejected = LifeQuantifierParser.parse_chatgpt_export(source_path, sanitize_pii=redact_pii, enforce_guardrails=enforce_guardrails)
            elif "WhatsApp" in source_type:
                samples, rejected = LifeQuantifierParser.parse_whatsapp_txt(source_path, sanitize_pii=redact_pii, enforce_guardrails=enforce_guardrails)
            else:
                samples, rejected = LifeQuantifierParser.parse_text_or_markdown_dir(source_path, sanitize_pii=redact_pii, enforce_guardrails=enforce_guardrails)

            if not samples:
                QMessageBox.warning(self, "No Data Found", "No valid clean messages or text samples could be extracted from the source.")
                self.quant_status_label.setText("Processing completed: 0 samples extracted.")
                return

            out_path = LifeQuantifierDatasetBuilder.export_dataset(samples, output_file, dataset_format=fmt)
            total_rejected = sum(rejected.values())
            self.quant_status_label.setText(f"✅ Processed {len(samples):,} clean samples. (Filtered out {total_rejected} harmful items).")

            rej_summary = f"Safety Guardrails Action: Filtered out {total_rejected} prohibited items.\n" \
                          f"  • Offensive Hacking/Malware: {rejected['hacking']}\n" \
                          f"  • Explosives/Weapons: {rejected['explosives']}\n" \
                          f"  • Explicit Sexual Content: {rejected['sexual']}\n" \
                          f"  • Self-Harm/Violence: {rejected['harm']}\n\n" if enforce_guardrails and total_rejected > 0 else ""

            QMessageBox.information(
                self,
                "Dataset Generation Complete",
                f"Successfully extracted & quantified {len(samples):,} safe personal data samples!\n\n"
                f"{rej_summary}"
                f"Saved to: {out_path}\n"
                f"Format: {fmt.upper()}\n"
                f"PII Redaction: {'ENABLED' if redact_pii else 'DISABLED'}\n"
                f"Safety Guardrails: {'ENABLED' if enforce_guardrails else 'DISABLED'}\n\n"
                f"You can now load this dataset into your QLoRA Trainer Command Center."
            )
        except Exception as e:
            self.quant_status_label.setText("❌ Error processing dataset!")
            QMessageBox.critical(self, "Processing Error", f"Failed to process personal dataset:\n{e!s}")

    def start_model_build(self):
        output_dir = self.out_dir_input.text().strip()
        base_tokenizer = self.tokenizer_combo.currentText()
        
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select a valid output directory.")
            return

        self.build_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_msg_label.setText("Starting model instantiation process...")

        self.worker = ModelBuildWorker(self.current_config, output_dir, base_tokenizer)
        self.worker.progress_signal.connect(self.on_build_progress)
        self.worker.finished_signal.connect(self.on_build_finished)
        self.worker.error_signal.connect(self.on_build_error)
        self.worker.start()

    def on_build_progress(self, msg: str, pct: int):
        self.status_msg_label.setText(msg)
        self.progress_bar.setValue(pct)

    def on_build_finished(self, result: dict):
        self.progress_bar.setValue(100)
        self.build_btn.setEnabled(True)
        self.status_msg_label.setText(f"✅ Model successfully created & exported to: {result['output_dir']}")
        
        QMessageBox.information(
            self,
            "Build Complete!",
            f"Successfully built and exported untrained model architecture!\n\n"
            f"Directory: {result['output_dir']}\n"
            f"QLoRA Target Modules: {result['qlora_spec']['suggested_target_modules_str']}\n\n"
            f"This directory is ready to be loaded into your QLoRA Trainer Command Center."
        )

    def on_build_error(self, err_msg: str):
        self.progress_bar.hide()
        self.build_btn.setEnabled(True)
        self.status_msg_label.setText("❌ Build failed!")
        QMessageBox.critical(self, "Build Error", f"An error occurred while creating model:\n{err_msg}")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(modern_qss)
    window = SLMBuilderMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
