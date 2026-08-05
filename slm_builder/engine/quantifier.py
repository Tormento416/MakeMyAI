"""
Quantify My Life Dataset Engine.
Parses personal chat exports (ChatGPT, WhatsApp, Text/Notes, Markdown), cleans/sanitizes PII,
and formats data into Alpaca, ShareGPT, or OpenAI JSONL datasets for QLoRA training.
"""

import json
import os
import re
from typing import Any, ClassVar


class SafetyGuardrails:
    """
    Safety Guardrails Engine.
    Filters out illegal, harmful, or prohibited content (offensive hacking/malware,
    explosives/weapons, sexual content, and self-harm/violence).
    """
    HARMFUL_PATTERNS: ClassVar[dict[str, list[str]]] = {
        "hacking": [
            r'\b(?:keylogger|metasploit|buffer overflow|ddos attack|sql injection|ransomware|malware payload|zero-day exploit|reverse shell|privilege escalation)\b',
            r'\b(?:bypass auth|crack password|steal credentials|botnet|remote code execution|rat trojan)\b'
        ],
        "explosives": [
            r'\b(?:bomb recipe|make explosive|anfo|tnt synthesis|pipe bomb|c4 synthesis|detonator|ied assembly|fertilizer bomb)\b',
            r'\b(?:how to make gunpowder|chemical weapon|mustard gas synthesis|ricin recipe)\b'
        ],
        "sexual": [
            r'\b(?:explicit sexual|pornography|erotic roleplay|nude pictures|sexually explicit|nsfw content|erotica)\b'
        ],
        "harm": [
            r'\b(?:how to commit suicide|self-harm instructions|how to kill|murder plan|assassination|physical assault|torture method)\b'
        ]
    }

    @classmethod
    def contains_harmful_content(cls, text: str) -> tuple[bool, str | None]:
        if not text:
            return False, None

        text_lower = text.lower()
        for category, patterns in cls.HARMFUL_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    return True, category
        return False, None


class PIISanitizer:

    """Strips private personal information (emails, phones, API keys, SSNs, IPv4)."""
    PATTERNS: ClassVar[dict[str, str]] = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "api_key": r'\b(?:sk-[a-zA-Z0-9]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|ghp_[a-zA-Z0-9]{36})\b',
        "ip": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }

    @classmethod
    def sanitize(cls, text: str) -> str:
        if not text:
            return ""
        text = re.sub(cls.PATTERNS["email"], "[REDACTED_EMAIL]", text)
        text = re.sub(cls.PATTERNS["phone"], "[REDACTED_PHONE]", text)
        text = re.sub(cls.PATTERNS["ssn"], "[REDACTED_SSN]", text)
        text = re.sub(cls.PATTERNS["api_key"], "[REDACTED_API_KEY]", text)
        text = re.sub(cls.PATTERNS["ip"], "[REDACTED_IP]", text)
        return text


class LifeQuantifierParser:
    """Parses local personal documents & export files."""

    @staticmethod
    def parse_chatgpt_export(file_path: str, sanitize_pii: bool = True, enforce_guardrails: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Parses standard ChatGPT conversations.json export while enforcing safety guardrails."""
        samples = []
        rejected_counts = {"hacking": 0, "explosives": 0, "sexual": 0, "harm": 0}
        
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        for conv in data:
            messages = []
            mapping = conv.get("mapping", {})
            for node in mapping.values():
                msg = node.get("message")
                if msg and msg.get("content") and msg.get("content").get("parts"):
                    role = msg.get("author", {}).get("role")
                    parts = msg.get("content", {}).get("parts", [])
                    content_str = "".join([p for p in parts if isinstance(p, str)]).strip()
                    if content_str and role in ["user", "assistant"]:
                        if enforce_guardrails:
                            is_harmful, category = SafetyGuardrails.contains_harmful_content(content_str)
                            if is_harmful:
                                rejected_counts[category] += 1
                                continue
                        if sanitize_pii:
                            content_str = PIISanitizer.sanitize(content_str)
                        messages.append({"role": role, "content": content_str})
            if len(messages) >= 2:
                samples.append({"conversations": messages})
        return samples, rejected_counts

    @staticmethod
    def parse_whatsapp_txt(file_path: str, sanitize_pii: bool = True, enforce_guardrails: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Parses WhatsApp export .txt file into multi-turn conversations while enforcing safety guardrails."""
        samples = []
        rejected_counts = {"hacking": 0, "explosives": 0, "sexual": 0, "harm": 0}
        pattern = re.compile(r'^\[?(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s?[AP]?M?)\]?\s([^:]+):\s(.*)$')
        
        current_messages = []
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    _, sender, content = match.groups()
                    if content and not content.startswith("<Media omitted>"):
                        if enforce_guardrails:
                            is_harmful, category = SafetyGuardrails.contains_harmful_content(content)
                            if is_harmful:
                                rejected_counts[category] += 1
                                continue
                        if sanitize_pii:
                            content = PIISanitizer.sanitize(content)
                        role = "user" if len(current_messages) % 2 == 0 else "assistant"
                        current_messages.append({"role": role, "content": f"{sender}: {content}"})
                        if len(current_messages) >= 6:
                            samples.append({"conversations": list(current_messages)})
                            current_messages = []
        if len(current_messages) >= 2:
            samples.append({"conversations": current_messages})
        return samples, rejected_counts

    @staticmethod
    def parse_text_or_markdown_dir(dir_path: str, chunk_size: int = 500, sanitize_pii: bool = True, enforce_guardrails: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Parses local text files or notes into QA / Alpaca instructions while enforcing safety guardrails."""
        samples = []
        rejected_counts = {"hacking": 0, "explosives": 0, "sexual": 0, "harm": 0}
        valid_exts = [".txt", ".md", ".json"]
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in valid_exts):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, encoding="utf-8", errors="ignore") as f:
                            text = f.read().strip()
                            if enforce_guardrails:
                                is_harmful, category = SafetyGuardrails.contains_harmful_content(text)
                                if is_harmful:
                                    rejected_counts[category] += 1
                                    continue
                            if sanitize_pii:
                                text = PIISanitizer.sanitize(text)
                            if text:
                                words = text.split()
                                for i in range(0, len(words), chunk_size):
                                    chunk = " ".join(words[i:i + chunk_size])
                                    if len(chunk) > 50:
                                        samples.append({
                                            "instruction": f"Elaborate on the personal notes regarding: {file}",
                                            "input": "",
                                            "output": chunk
                                        })
                    except Exception:
                        continue
        return samples, rejected_counts



class LifeQuantifierDatasetBuilder:
    """Builds and exports quantized personal life datasets for QLoRA fine-tuning."""

    @staticmethod
    def export_dataset(
        samples: list[dict[str, Any]],
        output_file: str,
        dataset_format: str = "sharegpt"  # sharegpt, alpaca, jsonl
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        formatted_rows = []
        for item in samples:
            if dataset_format == "sharegpt":
                if "conversations" in item:
                    formatted_rows.append(item)
                elif "instruction" in item:
                    formatted_rows.append({
                        "conversations": [
                            {"role": "user", "content": item["instruction"]},
                            {"role": "assistant", "content": item["output"]}
                        ]
                    })
            elif dataset_format == "alpaca":
                if "instruction" in item:
                    formatted_rows.append(item)
                elif "conversations" in item:
                    convs = item["conversations"]
                    if len(convs) >= 2:
                        formatted_rows.append({
                            "instruction": convs[0]["content"],
                            "input": "",
                            "output": convs[1]["content"]
                        })
            else:  # Raw JSONL
                formatted_rows.append(item)

        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in formatted_rows)

        return output_file
