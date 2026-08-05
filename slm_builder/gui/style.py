"""
Modern Dark Theme Styling Tokens for PySide6
"""

modern_qss = """
QWidget {
    background-color: #12141c;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #2d3748;
    background-color: #1a1d29;
    border-radius: 8px;
}

QTabBar::tab {
    background-color: #161923;
    color: #a0aec0;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 4px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #1a1d29;
    color: #6366f1;
    border-bottom: 2px solid #6366f1;
}

QGroupBox {
    border: 1px solid #2d3748;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
    color: #818cf8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0f1117;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f3f4f6;
}

QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #6366f1;
}

QPushButton {
    background-color: #4f46e5;
    color: #ffffff;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #4338ca;
}

QPushButton:pressed {
    background-color: #3730a3;
}

QPushButton#secondary_btn {
    background-color: #374151;
    color: #f3f4f6;
}

QPushButton#secondary_btn:hover {
    background-color: #4b5563;
}

QProgressBar {
    border: 1px solid #374151;
    border-radius: 6px;
    text-align: center;
    background-color: #0f1117;
    color: #ffffff;
}

QProgressBar::chunk {
    background-color: #6366f1;
    border-radius: 5px;
}

QTextEdit, QPlainTextEdit {
    background-color: #0f1117;
    border: 1px solid #2d3748;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    color: #a7f3d0;
}
"""

