#!/usr/bin/env python3
"""Generate and exercise a deterministic 5K dashboard vision workload.

The image deliberately mixes large visual structure with small operational text
so a passing result means more than recognizing a dominant color.  The second
request resends the complete conversation, including the original image, to
exercise OpenAI-compatible multi-turn semantics.
"""

import argparse
import base64
import io
import json
import os
import re
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 5120, 2880


def font(size, bold=False):
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
         else "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def dashboard_png(path):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#08111f")
    draw = ImageDraw.Draw(image)
    white, muted = "#f4f7fb", "#91a4bc"
    cyan, green, orange, purple = "#3fd8ff", "#45e39a", "#ffad42", "#b681ff"

    draw.rounded_rectangle((120, 100, 5000, 2780), 34, fill="#101c2d",
                           outline="#31445e", width=5)
    draw.text((220, 180), "ORBITAL RELAY CONTROL", font=font(86, True),
              fill=white)
    draw.text((220, 290), "Station AURORA-7  /  Deep-space operations",
              font=font(42), fill=muted)
    draw.ellipse((4610, 190, 4670, 250), fill=green)
    draw.text((4700, 190), "LIVE", font=font(44, True), fill=green)

    cards = [
        ("STATUS", "NOMINAL", green),
        ("ACTIVE LINKS", "12", cyan),
        ("QUEUE", "37", orange),
        ("UPLINK", "8.4 Gbps", purple),
    ]
    x = 220
    for label, value, color in cards:
        draw.rounded_rectangle((x, 410, x + 1110, 710), 24, fill="#17263a")
        draw.text((x + 55, 465), label, font=font(34, True), fill=muted)
        draw.text((x + 55, 545), value, font=font(68, True), fill=color)
        x += 1190

    panels = [
        (220, 790, 1650, 1770, "THERMAL GRID"),
        (1740, 790, 3460, 1770, "INCIDENT TIMELINE"),
        (3550, 790, 4780, 1770, "MAINTENANCE"),
    ]
    for x1, y1, x2, y2, title in panels:
        draw.rounded_rectangle((x1, y1, x2, y2), 24, fill="#132237",
                               outline="#293c57", width=3)
        draw.text((x1 + 55, y1 + 55), title, font=font(42, True), fill=white)

    thermal = [("GPU-0", "62 C", 0.62), ("GPU-1", "64 C", 0.64),
               ("COOLANT", "21 C", 0.32), ("PUMP", "73%", 0.73)]
    y = 940
    for label, value, frac in thermal:
        draw.text((285, y), label, font=font(34, True), fill=muted)
        draw.text((1290, y), value, font=font(36, True), fill=white)
        draw.rounded_rectangle((285, y + 65, 1445, y + 105), 14,
                               fill="#22334a")
        draw.rounded_rectangle((285, y + 65, 285 + int(1160 * frac),
                                y + 105), 14, fill=cyan if frac < .7 else orange)
        y += 190

    events = [
        ("09:42", "Packet loss spike", orange),
        ("09:47", "Reroute via Vega", purple),
        ("09:53", "Link restored", green),
    ]
    y = 970
    for stamp, text, color in events:
        draw.ellipse((1810, y, 1850, y + 40), fill=color)
        draw.line((1830, y + 40, 1830, y + 210), fill="#43566f", width=5)
        draw.text((1890, y - 8), stamp, font=font(38, True), fill=color)
        draw.text((2110, y - 8), text, font=font(40), fill=white)
        y += 235

    maintenance = [
        ("TASK", "Replace filter unit B-12"),
        ("INSPECTION DUE", "2031-04-18"),
        ("OWNER", "Mira Chen"),
        ("PRIORITY", "MEDIUM"),
    ]
    y = 970
    for label, value in maintenance:
        draw.text((3620, y), label, font=font(28, True), fill=muted)
        draw.text((3620, y + 48), value, font=font(38, True),
                  fill=orange if label == "PRIORITY" else white)
        y += 180

    draw.rounded_rectangle((220, 1860, 4780, 2520), 24, fill="#132237",
                           outline="#293c57", width=3)
    draw.text((285, 1915), "TRAFFIC — LAST 60 MINUTES", font=font(42, True),
              fill=white)
    left, top, right, bottom = 400, 2100, 4590, 2410
    for i in range(6):
        yy = top + i * (bottom - top) // 5
        draw.line((left, yy, right, yy), fill="#263950", width=2)
    ingress = [(left + i * 380, top + v) for i, v in enumerate(
        [250, 220, 235, 170, 190, 110, 130, 85, 120, 60, 90, 40])]
    egress = [(left + i * 380, top + v) for i, v in enumerate(
        [280, 260, 240, 250, 200, 215, 175, 190, 155, 170, 120, 135])]
    draw.line(ingress, fill=purple, width=14, joint="curve")
    draw.line(egress, fill=orange, width=14, joint="curve")
    draw.rectangle((3800, 1940, 3850, 1990), fill=purple)
    draw.text((3870, 1937), "Ingress", font=font(32), fill=muted)
    draw.rectangle((4200, 1940, 4250, 1990), fill=orange)
    draw.text((4270, 1937), "Egress", font=font(32), fill=muted)

    footer = ("SECURITY CODE  COBALT-917     REGION  CA-EAST-4     "
              "BUILD  6.12.3     LAST SYNC  09:58:21 UTC")
    draw.text((260, 2615), footer, font=font(30, True), fill="#7890aa")
    image.save(path, format="PNG", optimize=True)
    return image


def post(base, key, model, messages, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body, headers=headers, method="POST")
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def message_text(result):
    content = result["choices"][0]["message"].get("content") or ""
    if isinstance(content, list):
        return "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x)
                         for x in content)
    return str(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image = dashboard_png(args.image)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", optimize=True)
    data_uri = ("data:image/png;base64,"
                + base64.b64encode(encoded.getvalue()).decode())
    first_user = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": (
            "Describe this operations dashboard comprehensively and precisely. "
            "Include its title and station, all four summary cards, thermal "
            "readings, the incident timeline, maintenance task/date/owner, "
            "traffic line colors, and the small footer metadata.")},
    ]}
    first, first_s = post(args.base_url, args.api_key, args.model,
                          [first_user], 1600)
    first_text = message_text(first)
    follow, follow_s = post(args.base_url, args.api_key, args.model, [
        first_user,
        {"role": "assistant", "content": first_text},
        {"role": "user", "content": (
            "From the same screenshot, reply with only the security code, "
            "maintenance owner, and link-restoration time, separated by pipes.")},
    ], 100)
    follow_text = message_text(follow)
    text_only, text_s = post(args.base_url, args.api_key, args.model, [
        {"role": "user", "content": "Reply with exactly VISION_TEXT_OK."},
    ], 32)
    text_only_text = message_text(text_only)

    required = [
        "aurora-7", "nominal", "12", "37", "8.4", "62", "64", "21",
        "73", "09:42", "09:47", "09:53", "b-12", "2031-04-18",
        "mira chen", "cobalt-917", "ca-east-4", "6.12.3",
    ]

    def term_matched(term, text):
        # Bare numbers self-match inside other required values ("12" inside
        # "b-12" or "6.12.3") and silently eat the 2-miss budget the >=16
        # threshold represents. Anchor purely numeric terms on
        # non-alphanumeric/dot/dash boundaries; compound terms keep plain
        # substring search.
        if re.fullmatch(r"[0-9.:]+", term):
            return re.search(r"(?<![0-9a-z.-])" + re.escape(term)
                             + r"(?![0-9a-z.-])", text) is not None
        return term in text

    lower = first_text.lower()
    checks = {
        "image_dimensions_5k": image.size == (5120, 2880),
        "first_turn_detail_recall": {
            "passed": sum(term_matched(term, lower) for term in required) >= 16,
            "matched": [term for term in required if term_matched(term, lower)],
            "required": required,
        },
        "multi_turn_image_recall": all(
            term in follow_text.lower()
            for term in ("cobalt-917", "mira chen", "09:53")),
        "text_only_regression": "VISION_TEXT_OK" in text_only_text,
    }
    document = {
        "schema": 1,
        "base_url": args.base_url,
        "model": args.model,
        "image": {"path": args.image, "width": WIDTH, "height": HEIGHT,
                  "bytes": os.path.getsize(args.image)},
        "timings_s": {"vision_first": first_s, "vision_followup": follow_s,
                      "text_only": text_s},
        "checks": checks,
        "responses": {"vision_first": first_text,
                      "vision_followup": follow_text,
                      "text_only": text_only_text},
        "usage": {"vision_first": first.get("usage"),
                  "vision_followup": follow.get("usage"),
                  "text_only": text_only.get("usage")},
    }
    with open(args.out, "w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    print(json.dumps(checks, indent=2))
    return 0 if (
        checks["first_turn_detail_recall"]["passed"]
        and checks["multi_turn_image_recall"]
        and checks["text_only_regression"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
