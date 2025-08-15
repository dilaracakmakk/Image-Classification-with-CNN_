import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageOps, ImageFile
import os, json, re, pandas as pd


AUG_PATH = "augmented_dataset"         
CLASS_INDEX_JSON = "class_indices.json"
CSV_FILE = "model_kodlari.csv"          
DEFAULT_IMAGE_PATH = "emirali.jpg"      

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE, delimiter=';', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    for col in ["Model No", "Firma Model No", "Giysi Grubu", "Giysi Cinsi"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
else:
    df = pd.DataFrame(columns=["Model No","Firma Model No","Giysi Grubu","Giysi Cinsi"])  

if os.path.exists(CLASS_INDEX_JSON):
    with open(CLASS_INDEX_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
        CATEGORIES = [raw[str(i)] for i in range(len(raw))]
else:
    CATEGORIES = [d for d in os.listdir(AUG_PATH) if os.path.isdir(os.path.join(AUG_PATH, d))] if os.path.exists(AUG_PATH) else ["dress","pants","tshirt"]

ImageFile.LOAD_TRUNCATED_IMAGES = True  
THUMB_SIZE = (120, 120)
THUMB_CACHE_DIR = "_thumb_cache"
os.makedirs(THUMB_CACHE_DIR, exist_ok=True)
CAT_DIR_CACHE = {} 

def load_category_json(path="category_info.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

CATEGORY_INFO = load_category_json()

def extract_model_no_from_filename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    name = re.sub(r"\(.*?\)", "", name)  
    name = name.split("_")[0]              
    return name.strip().lower()

def show_csv_info(file_path):
    model_no_clean = extract_model_no_from_filename(file_path)
    try:
        matched = df[(df.get("Firma Model No", "") == model_no_clean) | (df.get("Model No", "") == model_no_clean)] if not df.empty else pd.DataFrame()
    except Exception:
        matched = pd.DataFrame()

    if not matched.empty:
        row = matched.iloc[0]
        update_info_panel(
            row.get("Model No", "-"),
            row.get("Firma Model No", "-"),
            row.get("Giysi Grubu", "-"),
            row.get("Giysi Cinsi", "-")
        )
    else:
        fn = os.path.basename(file_path)
        cat = "-"
        meta = CATEGORY_INFO.get(fn)
        if isinstance(meta, (list, tuple)) and len(meta) == 2:
            cat = meta[1]
        update_info_panel(model_no_clean, "-", "-", cat)


def on_thumb_click(path):
    global current_idx, current_image_path
    current_image_path = path
    ensure_gallery_for_path(path)
    try:
        current_idx = gallery_filepaths.index(path)
    except ValueError:
        current_idx = 0
    try:
        gallery_frame.grid_remove()
    except Exception:
        pass
    img = Image.open(path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    img_label.grid()
    show_csv_info(path)



def _ensure_category_listing(category):
    wanted = category.lower().rstrip('s')
    if wanted in CAT_DIR_CACHE:
        return CAT_DIR_CACHE[wanted]
    if not os.path.exists(AUG_PATH):
        return None, []
    candidates = [d for d in os.listdir(AUG_PATH) if os.path.isdir(os.path.join(AUG_PATH, d))]
    selected_folder = None
    for d in candidates:
        if d.lower().rstrip('s') == wanted:
            selected_folder = d
            break
    if not selected_folder:
        return None, []
    folder_path = os.path.join(AUG_PATH, selected_folder)
    file_list = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith((".jpg",".jpeg",".png"))]
    CAT_DIR_CACHE[wanted] = (selected_folder, file_list)
    return selected_folder, file_list


def _get_cached_thumb(src_path):
    try:
        rel = os.path.relpath(src_path, AUG_PATH)
        dest = os.path.join(THUMB_CACHE_DIR, os.path.splitext(rel)[0] + ".jpg")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if (not os.path.exists(dest)) or (os.path.getmtime(dest) < os.path.getmtime(src_path)):
            with Image.open(src_path) as im:
                im = ImageOps.exif_transpose(im)
                im.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
                im.save(dest, "JPEG", quality=80, optimize=True)
        return dest
    except Exception:
        return src_path


def display_images_by_category(category):
    """Seçilen kategori için tüm görselleri solda, kaydırılabilir küçük kutucuklar halinde göster.
       Performans: klasör/file listesi cache, disk thumbnail cache, batch render.
    """
    global gallery_filepaths, current_idx, last_selected_category
    try:
        img_label.grid_remove()
    except Exception:
        pass
    gallery_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nwse")

    for w in gallery_inner.winfo_children():
        w.destroy()
    gallery_images_refs.clear()
    gallery_filepaths = []
    current_idx = -1

    selected_folder, file_list = _ensure_category_listing(category)
    if not selected_folder:
        return
    last_selected_category = selected_folder
    gallery_filepaths = file_list

    BATCH = 80
    def _render_batch(start_idx=0):
        end_idx = min(len(gallery_filepaths), start_idx + BATCH)
        for i in range(start_idx, end_idx):
            p = gallery_filepaths[i]
            tp = _get_cached_thumb(p)
            try:
                with Image.open(tp) as im:
                    img_tk = ImageTk.PhotoImage(im)
                gallery_images_refs.append(img_tk)
                r, c = divmod(i, 5)
                thumb = tk.Label(gallery_inner, image=img_tk, bg="white", bd=1, relief="solid", cursor="hand2")
                thumb.image = img_tk
                thumb.grid(row=r, column=c, padx=6, pady=6)
                thumb.bind("<Button-1>", lambda e, p=p: on_thumb_click(p))
            except Exception as e:
                print("Thumb yüklenemedi:", e)
        gallery_inner.update_idletasks()
        gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all"))
        if end_idx < len(gallery_filepaths):
            root.after(1, lambda: _render_batch(end_idx))

    _render_batch(0)

    for w in gallery_inner.winfo_children():
        w.destroy()
    gallery_images_refs.clear()

    wanted = category.lower().rstrip('s')

    if not os.path.exists(AUG_PATH):
        return

    candidates = [d for d in os.listdir(AUG_PATH) if os.path.isdir(os.path.join(AUG_PATH, d))]
    selected_folder = None
    for d in candidates:
        if d.lower().rstrip('s') == wanted:
            selected_folder = d
            break
    if not selected_folder:
        return

    folder_path = os.path.join(AUG_PATH, selected_folder)

    row = col = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg",".jpeg",".png")):
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path).resize((110, 110))
                img_tk = ImageTk.PhotoImage(img)
                thumb = tk.Label(gallery_inner, image=img_tk, bg="white", bd=1, relief="solid", cursor="hand2")
                thumb.image = img_tk
                gallery_images_refs.append(img_tk)
                thumb.grid(row=row, column=col, padx=6, pady=6)
                thumb.bind("<Button-1>", lambda e, p=image_path: on_thumb_click(p))
                col += 1
                if col >= 5:
                    col = 0
                    row += 1
            except Exception as e:
                print("Görsel yüklenemedi:", e)

    gallery_inner.update_idletasks()
    gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all"))


def upload_and_predict():
    global current_image_path
    path = filedialog.askopenfilename(filetypes=[("Görsel", "*.jpg *.jpeg *.png")])
    if path:
        current_image_path = path
        on_thumb_click(path)


def open_filter_panel():
    win = tk.Toplevel(root)
    win.title("Gelişmiş Filtreleme")
    win.geometry("320x220+980+120")

    ttk.Label(win, text="Kategori").pack(anchor="w", padx=10, pady=(10,4))
    cat_combo = ttk.Combobox(win, values=CATEGORIES, state="readonly")
    cat_combo.pack(padx=10, fill="x")

    def do_search():
        sel = cat_combo.get().strip()
        if sel:
            display_images_by_category(sel)
    ttk.Button(win, text="Ara", command=do_search).pack(pady=12)

root = tk.Tk()
root.title("Kıyafet Tanımlama Paneli")
root.geometry("1000x760")
root.configure(bg="#f6f7fb")

header = tk.Label(root, text="Kıyafet Tanımlama Paneli", font=("Segoe UI", 22, "bold"), bg="#f6f7fb")
header.pack(pady=16)

content_frame = tk.Frame(root, bg="white", bd=1, relief="solid")
content_frame.pack(padx=20, pady=10, fill="both", expand=True)
content_frame.grid_rowconfigure(0, weight=1)
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=0)
content_frame.grid_columnconfigure(2, weight=0)

img_label = tk.Label(content_frame, bg="#ffffff")
img_label.grid(row=0, column=0, padx=20, pady=(20,6), sticky="nw")

left_btn = tk.Button(content_frame, text="◀", command=lambda: navigate(-1), width=3)
left_btn.grid(row=1, column=0, padx=20, pady=(0,12), sticky="w")
right_btn = tk.Button(content_frame, text="▶", command=lambda: navigate(1), width=3)
right_btn.grid(row=1, column=0, padx=20, pady=(0,12), sticky="e")

gallery_frame = tk.Frame(content_frame, bg="#ffffff")
gallery_canvas = tk.Canvas(gallery_frame, bg="#ffffff", highlightthickness=0, width=520, height=460)
scrollbar = ttk.Scrollbar(gallery_frame, orient="vertical", command=gallery_canvas.yview)

gallery_inner = tk.Frame(gallery_canvas, bg="#ffffff")
inner_window = gallery_canvas.create_window((0,0), window=gallery_inner, anchor="nw")

gallery_canvas.configure(yscrollcommand=scrollbar.set)

gallery_canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")
gallery_frame.grid_rowconfigure(0, weight=1)
gallery_frame.grid_columnconfigure(0, weight=1)

gallery_images_refs = [] 
gallery_filepaths = []
current_idx = -1
last_selected_category = None 
current_image_path = None      

def ensure_gallery_for_path(path):
    """If gallery list does not match the folder of path, rebuild it from that folder."""
    global gallery_filepaths, last_selected_category
    folder = os.path.dirname(path)
    last_selected_category = os.path.basename(folder)
    if not gallery_filepaths or os.path.dirname(gallery_filepaths[0]) != folder:
        files = []
        try:
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith((".jpg",".jpeg",".png")):
                    files.append(os.path.join(folder, fn))
        except Exception:
            files = [path]
        gallery_filepaths = files
    return
    if current_idx == -1:
        current_idx = 0
    else:
        current_idx = (current_idx + delta) % len(gallery_filepaths)
    on_thumb_click(gallery_filepaths[current_idx])


def navigate(delta):
    global current_idx
    if not gallery_filepaths:
        return
    if current_idx == -1:
        current_idx = 0
    else:
        current_idx = (current_idx + delta) % len(gallery_filepaths)
    on_thumb_click(gallery_filepaths[current_idx])

def on_left(_=None):
    navigate(-1)

def on_right(_=None):
    navigate(1)


def go_back():
    """Top (▲) button action: return to a gallery.
    Priority: last_selected_category (if exists in AUG_PATH), otherwise infer from current_image_path
    if it resides under AUG_PATH. If none available, do nothing.
    """
    if last_selected_category and os.path.isdir(os.path.join(AUG_PATH, last_selected_category)):
        display_images_by_category(last_selected_category)
        return
    if current_image_path:
        abs_aug = os.path.abspath(AUG_PATH)
        abs_cur = os.path.abspath(current_image_path)
        if abs_cur.startswith(abs_aug):
            cat = os.path.basename(os.path.dirname(abs_cur))
            if cat and os.path.isdir(os.path.join(AUG_PATH, cat)):
                display_images_by_category(cat)
                return


def _gal_conf(_):
    gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all"))

gallery_inner.bind("<Configure>", _gal_conf)

info_frame = tk.Frame(content_frame, bg="#ffffff")
info_frame.grid(row=0, column=1, padx=20, pady=20, sticky="ne")

info_title = tk.Label(info_frame, text="Ürün Bilgisi", font=("Segoe UI", 12, "bold"), bg="#ffffff")
info_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,8))

def _add_info_row(r, label_text):
    tk.Label(info_frame, text=label_text, font=("Segoe UI", 11, "bold"), bg="#ffffff").grid(row=r, column=0, sticky="w", padx=(0,10), pady=4)
    val = tk.Label(info_frame, text="-", font=("Segoe UI", 11), bg="#ffffff", anchor="w", width=28, wraplength=280, justify="left")
    val.grid(row=r, column=1, sticky="w", pady=4)
    return val

info_model_val = _add_info_row(1, "Model No:")
info_firma_val = _add_info_row(2, "Firma Model No:")
info_grup_val  = _add_info_row(3, "Giysi Grubu:")
info_cins_val  = _add_info_row(4, "Giysi Cinsi:")


def update_info_panel(model_no, firma_no, grup, cins):
    info_model_val.config(text=str(model_no) if model_no else "-")
    info_firma_val.config(text=str(firma_no) if firma_no else "-")
    info_grup_val.config(text=str(grup) if grup else "-")
    info_cins_val.config(text=str(cins) if cins else "-")

top_controls = tk.Frame(content_frame, bg="white")
up_btn = tk.Button(top_controls, text="▲", font=("Segoe UI", 12), command=go_back, bd=0, bg="white")
hamburger_btn = tk.Button(top_controls, text="☰", font=("Segoe UI", 14), command=open_filter_panel, bd=0, bg="white")
up_btn.pack(side="left", padx=(0,8))
hamburger_btn.pack(side="left")
top_controls.grid(row=0, column=2, padx=10, pady=10, sticky="ne")

select_btn = tk.Button(root, text="Görsel Seç", command=upload_and_predict,
                       bg="#1abc9c", fg="white", font=("Segoe UI", 12, "bold"), padx=20, pady=10, relief="flat")
select_btn.pack(pady=18)

if os.path.exists(DEFAULT_IMAGE_PATH):
    try:
        _img = Image.open(DEFAULT_IMAGE_PATH).resize((300, 300))
        _img_tk = ImageTk.PhotoImage(_img)
        img_label.config(image=_img_tk)
        img_label.image = _img_tk
    except Exception:
        pass

root.bind_all("<Left>", on_left)
root.bind_all("<Right>", on_right)

root.mainloop()
