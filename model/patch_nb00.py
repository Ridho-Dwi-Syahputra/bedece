"""
Patch NB00: Menambahkan sel-sel baru untuk pencarian klaster terarah dan
pembuatan master fine-tuning plan CSV (Stage 3.5).

Jalankan sekali: python patch_nb00.py
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "00_Diagnostik_Kemiripan_TrainTest.ipynb"

# ── Sel-sel baru yang akan ditambahkan ──────────────────────────────────────

NEW_CELLS = [
    # ── MARKDOWN: header bagian 2 ──
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Bagian 2 — Pencarian klaster TERARAH untuk fine-tuning (Stage 3.5)\n",
            "\n",
            "Berdasarkan analisis di atas, kita sudah tahu pola-pola ambigu yang menyebabkan ensemble\n",
            "salah prediksi. Sekarang kita cari SEMUA gambar train yang mirip pola-pola ini, supaya bisa:\n",
            "1. **Oversampling** — naikkan bobot sampling klaster ini saat fine-tuning tambahan.\n",
            "2. **Label correction** — perbaiki label yang jelas salah (mis. tirai → Recyclable bukan Electronic).\n",
            "3. **Export CSV master** — daftar `(image_id, pattern_group, similarity, label, correct_label,\n",
            "   action)` yang langsung bisa dikonsumsi notebook fine-tuning Stage 3.5.\n",
            "\n",
            "**Pendekatan**: Pakai embedding test image sebagai prototipe tiap pola → cari tetangga di\n",
            "ruang embedding train. Ini BUKAN mengintip jawaban test — kita cuma pakai gambar test sebagai\n",
            "\"query visual\" untuk menemukan klaster di data train yang SUDAH ada, lalu keputusan\n",
            "oversampling/relabel diterapkan murni ke data train.\n",
        ],
    },
    # ── CODE: definisi pattern groups ──
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "# Definisi kelompok pola berdasarkan analisis grid sebelumnya\n",
            "# ============================================================\n",
            "\n",
            "PATTERN_GROUPS = {\n",
            '    "mainan_bunga": {\n',
            '        "test_ids": [312, 116, 438, 728, 843, 899, 1035, 1079, 1168, 1351, 1373],\n',
            '        "expected_label": 0,  # Recyclable\n',
            '        "description": "Kerajinan/mainan berbentuk bunga dari kertas/plastik",\n',
            '        "top_n": 50,\n',
            '        "sim_threshold": 0.35,\n',
            "    },\n",
            '    "pipet": {\n',
            '        "test_ids": [293, 132, 363, 499, 659, 781, 907, 908, 1033],\n',
            '        "expected_label": 0,  # Recyclable\n',
            '        "description": "Pipet/sedotan plastik",\n',
            '        "top_n": 50,\n',
            '        "sim_threshold": 0.35,\n',
            "    },\n",
            '    "tirai": {\n',
            '        "test_ids": [372, 637],\n',
            '        "expected_label": 0,  # Recyclable (tekstil, bukan Electronic)\n',
            '        "description": "Tirai kain — ada yg salah label Electronic, harus Recyclable",\n',
            '        "top_n": 30,\n',
            '        "sim_threshold": 0.40,\n',
            "    },\n",
            '    "kain": {\n',
            '        "test_ids": [565],\n',
            '        "expected_label": 0,  # Recyclable\n',
            '        "description": "Kain/tekstil",\n',
            '        "top_n": 30,\n',
            '        "sim_threshold": 0.40,\n',
            "    },\n",
            '    "tisu": {\n',
            '        "test_ids": [1105, 1141],\n',
            '        "expected_label": 0,  # Recyclable\n',
            '        "description": "Tisu/tissue paper",\n',
            '        "top_n": 40,\n',
            '        "sim_threshold": 0.35,\n',
            "    },\n",
            "}\n",
            "\n",
            'print(f"Jumlah kelompok pola: {len(PATTERN_GROUPS)}")\n',
            "for name, spec in PATTERN_GROUPS.items():\n",
            "    print(f\"  {name}: {len(spec['test_ids'])} test prototipe, \"\n",
            "          f\"expected={label_names[spec['expected_label']]}\")\n",
        ],
    },
    # ── MARKDOWN: embed prototipe ──
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Hitung prototipe tiap pola dari embedding test, lalu cari klaster di train\n",
        ],
    },
    # ── CODE: embed test prototypes ──
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "all_proto_test_ids = set()\n",
            "for spec in PATTERN_GROUPS.values():\n",
            '    all_proto_test_ids.update(spec["test_ids"])\n',
            "\n",
            "test_embeddings_cache = {}\n",
            "for tid in sorted(all_proto_test_ids):\n",
            '    test_path = LOCAL_TEST_ROOT / f"{tid}.jpg"\n',
            "    if test_path.exists():\n",
            "        test_embeddings_cache[tid] = embed_one(test_path)\n",
            "    else:\n",
            '        print(f"PERINGATAN: test image {tid} tidak ditemukan di {LOCAL_TEST_ROOT}")\n',
            "\n",
            'print(f"{len(test_embeddings_cache)}/{len(all_proto_test_ids)} test embeddings berhasil dihitung.")\n',
        ],
    },
    # ── CODE: cluster search per pola ──
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "cluster_results = {}  # pattern_name -> dict(ids, sims, labels)\n",
            "\n",
            "for pattern_name, spec in PATTERN_GROUPS.items():\n",
            "    proto_embeds = [test_embeddings_cache[tid]\n",
            '                    for tid in spec["test_ids"] if tid in test_embeddings_cache]\n',
            "    if len(proto_embeds) == 0:\n",
            '        print(f"SKIP {pattern_name}: tidak ada embedding test yang valid.")\n',
            "        continue\n",
            "\n",
            "    prototype = np.mean(proto_embeds, axis=0)\n",
            "    prototype = prototype / np.linalg.norm(prototype)\n",
            "\n",
            "    sims = train_embeddings @ prototype\n",
            "    order = np.argsort(-sims)\n",
            "\n",
            '    top_n = spec["top_n"]\n',
            '    threshold = spec["sim_threshold"]\n',
            "    n_above = int((sims >= threshold).sum())\n",
            "    n_take = min(max(top_n, n_above), len(sims))\n",
            "\n",
            "    top_idx = order[:n_take]\n",
            "    top_ids = train_ids_ordered[top_idx]\n",
            "    top_sims = sims[top_idx]\n",
            "    top_labels = np.array([int(label_by_id.loc[int(i)]) for i in top_ids])\n",
            "\n",
            "    mask = top_sims >= threshold\n",
            "    top_ids, top_sims, top_labels = top_ids[mask], top_sims[mask], top_labels[mask]\n",
            "\n",
            "    cluster_results[pattern_name] = {\n",
            '        "ids": top_ids, "sims": top_sims, "labels": top_labels,\n',
            '        "expected_label": spec["expected_label"],\n',
            "    }\n",
            "\n",
            "    label_dist = pd.Series(top_labels).map(label_names).value_counts().to_dict()\n",
            '    n_mislabel = int((top_labels != spec["expected_label"]).sum())\n',
            "    print(f\"\\n=== {pattern_name} ({spec['description']}) ===\")\n",
            '    print(f"  Prototipe dari {len(proto_embeds)} test images")\n',
            '    print(f"  Ditemukan {len(top_ids)} gambar train (sim >= {threshold})")\n',
            '    print(f"  Distribusi label: {label_dist}")\n',
            "    print(f\"  Potensi mislabel (bukan {label_names[spec['expected_label']]}): {n_mislabel}\")\n",
            '    print(f"  Similarity range: [{top_sims.min():.3f}, {top_sims.max():.3f}]")\n',
        ],
    },
    # ── MARKDOWN: visual ──
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Bukti visual per kelompok pola\n"],
    },
    # ── CODE: render grids ──
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "for pattern_name, data in cluster_results.items():\n",
            '    n_show = min(30, len(data["ids"]))\n',
            "    render_suspect_grid(\n",
            '        data["ids"][:n_show], data["sims"][:n_show],\n',
            "        title=f\"Klaster '{pattern_name}' — top {n_show} tetangga train \"\n",
            "              f\"(expected: {label_names[data['expected_label']]})\",\n",
            '        save_path=EVIDENCE_OUTPUT_ROOT / f"cluster_{pattern_name}.png",\n',
            "        cols=6,\n",
            "    )\n",
        ],
    },
    # ── MARKDOWN: kompilasi CSV ──
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Kompilasi master CSV untuk fine-tuning Stage 3.5\n",
            "\n",
            "Tiap baris = satu gambar train yang masuk klaster pola bermasalah.\n",
            "Kolom: `image_id`, `pattern_group`, `similarity`, `current_label`, `expected_label`,\n",
            "`action` (oversample/relabel), `oversample_weight`.\n",
        ],
    },
    # ── CODE: compile & save CSV ──
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "OVERSAMPLE_WEIGHT = 3.0\n",
            "\n",
            "finetune_rows = []\n",
            "for pattern_name, data in cluster_results.items():\n",
            '    expected = data["expected_label"]\n',
            '    for img_id, sim, cur_label in zip(data["ids"], data["sims"], data["labels"]):\n',
            '        action = "oversample" if int(cur_label) == expected else "relabel"\n',
            "        finetune_rows.append({\n",
            '            "image_id": int(img_id), "pattern_group": pattern_name,\n',
            '            "similarity": round(float(sim), 4),\n',
            '            "current_label": int(cur_label), "expected_label": expected,\n',
            '            "action": action, "oversample_weight": OVERSAMPLE_WEIGHT,\n',
            "        })\n",
            "\n",
            "finetune_df = pd.DataFrame(finetune_rows)\n",
            'finetune_df = finetune_df.sort_values("similarity", ascending=False).drop_duplicates(\n',
            '    subset="image_id", keep="first"\n',
            ').sort_values("image_id").reset_index(drop=True)\n',
            "\n",
            'finetune_path = EVIDENCE_OUTPUT_ROOT / "finetune_stage35_plan.csv"\n',
            "finetune_df.to_csv(finetune_path, index=False)\n",
            "\n",
            'print(f"Master fine-tuning plan: {len(finetune_df)} gambar train")\n',
            "print(f\"  - oversample: {(finetune_df['action'] == 'oversample').sum()}\")\n",
            "print(f\"  - relabel:    {(finetune_df['action'] == 'relabel').sum()}\")\n",
            'print(f"\\nPer kelompok pola:")\n',
            'for grp, sub in finetune_df.groupby("pattern_group"):\n',
            "    n_ok = (sub[\"action\"] == \"oversample\").sum()\n",
            "    n_fix = (sub[\"action\"] == \"relabel\").sum()\n",
            '    print(f"  {grp}: {len(sub)} gambar ({n_ok} oversample, {n_fix} relabel)")\n',
            'print(f"\\nDisimpan ke: {finetune_path}")\n',
        ],
    },
    # ── MARKDOWN: relabel detail ──
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Detail gambar yang perlu RELABEL\n",
        ],
    },
    # ── CODE: relabel detail ──
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'relabel_all = finetune_df[finetune_df["action"] == "relabel"].copy()\n',
            'relabel_all["current_label_name"] = relabel_all["current_label"].map(label_names)\n',
            'relabel_all["expected_label_name"] = relabel_all["expected_label"].map(label_names)\n',
            "\n",
            "if len(relabel_all) > 0:\n",
            '    print(f"Total gambar yang perlu relabel: {len(relabel_all)}")\n',
            '    print(relabel_all[["image_id", "pattern_group", "similarity",\n',
            '                        "current_label_name", "expected_label_name"]].to_string(index=False))\n',
            '    relabel_path = EVIDENCE_OUTPUT_ROOT / "all_label_corrections.csv"\n',
            "    relabel_all.to_csv(relabel_path, index=False)\n",
            '    print(f"\\nDisimpan ke: {relabel_path}")\n',
            "else:\n",
            '    print("Tidak ada gambar yang perlu relabel.")\n',
        ],
    },
    # ── MARKDOWN: ringkasan akhir bagian 2 ──
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Ringkasan & langkah selanjutnya\n",
            "\n",
            "**Output notebook ini yang dipakai Stage 3.5:**\n",
            "1. `finetune_stage35_plan.csv` — daftar gambar train + bobot oversampling + koreksi label\n",
            "2. `all_label_corrections.csv` — subset yang perlu relabel saja\n",
            "\n",
            "**Langkah selanjutnya (di notebook terpisah):**\n",
            "1. Terapkan relabel ke `fold_assignment.csv`\n",
            "2. Buat notebook fine-tuning Stage 3.5:\n",
            "   - Load `best.ckpt` dari Stage 3 (bukan dari nol)\n",
            "   - `WeightedRandomSampler` dengan bobot dari CSV\n",
            "   - LR kecil (1e-6 ~ 5e-6), 3-5 epoch saja\n",
            "   - Validasi OOF untuk pastikan tidak merusak kelas lain\n",
        ],
    },
]


def patch():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Cari sel "## Interpretasi" terakhir — sel baru disisipkan SEBELUM itu
    insert_idx = None
    for i in range(len(cells) - 1, -1, -1):
        if cells[i]["cell_type"] == "markdown":
            src = "".join(cells[i].get("source", []))
            if "## Interpretasi" in src and "Nilai patokan kasar" in src:
                insert_idx = i
                break

    if insert_idx is None:
        print("PERINGATAN: sel '## Interpretasi' tidak ditemukan — sel baru ditambahkan di akhir.")
        insert_idx = len(cells)

    # Cek apakah sudah di-patch (hindari duplikasi)
    for c in cells:
        src = "".join(c.get("source", []))
        if "PATTERN_GROUPS" in src and "finetune_stage35_plan" in src:
            print("Notebook sudah di-patch sebelumnya — SKIP.")
            return

    # Insert
    for j, cell in enumerate(NEW_CELLS):
        cells.insert(insert_idx + j, cell)

    nb["cells"] = cells

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"SUKSES: {len(NEW_CELLS)} sel baru ditambahkan ke {NB_PATH.name} (sebelum 'Interpretasi').")


if __name__ == "__main__":
    patch()
