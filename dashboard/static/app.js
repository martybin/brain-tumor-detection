const state = {
  id: null,
  modalities: [],
  image: false,
  shape: null,
  axis: "axial",
  modality: "flair",
  index: 0,
  space: "original",
  overlay: 0,
};

const slots = ["t1", "t1ce", "t2", "flair"];
const axisIndex = { sagittal: 0, coronal: 1, axial: 2 };
const drop = document.querySelector("#drop");
const fileInput = document.querySelector("#files");

document.querySelector("#choose").addEventListener("click", () => fileInput.click());
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("hot"); });
drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("hot");
  uploadFiles(event.dataTransfer.files);
});
fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
document.querySelector("#sample").addEventListener("click", loadSample);
document.querySelector("#predict").addEventListener("click", predict);
document.querySelector("#index").addEventListener("input", (event) => {
  state.index = Number(event.target.value);
  showSlice();
});
document.querySelector("#axes").addEventListener("click", (event) => {
  const axis = event.target.dataset.axis;
  if (!axis) return;
  state.axis = axis;
  document.querySelectorAll("#axes button").forEach((button) => button.classList.toggle("on", button.dataset.axis === axis));
  resetIndex();
  showSlice();
});

boot();

async function boot() {
  const health = await fetch("/api/health").then((response) => response.json());
  document.querySelector("#disclaimer").textContent = health.disclaimer;
  document.querySelector("#model-state").textContent = health.checkpoint_ready
    ? "چک‌پوینت آماده است"
    : "مدل هنوز آموزش ندیده";
  renderSlots([]);
}

function renderSlots(present) {
  document.querySelector("#slots").innerHTML = slots.map((name) => {
    const ok = present.includes(name);
    return `<li><span>${name}</span><em class="${ok ? "" : "missing"}">${ok ? "حاضر" : "نیست"}</em></li>`;
  }).join("");
}

function classify(file) {
  const name = file.name.toLowerCase();
  if (/\.(png|jpe?g|webp)$/.test(name)) return "picture";
  if (!name.endsWith(".nii") && !name.endsWith(".nii.gz")) return null;
  if (name.includes("t1ce") || name.includes("t1c") || name.includes("t1gd")) return "t1ce";
  if (name.includes("flair") || name.includes("t2f")) return "flair";
  if (name.includes("t2")) return "t2";
  if (name.includes("t1")) return "t1";
  return null;
}

async function uploadFiles(fileList) {
  const body = new FormData();
  let count = 0;
  for (const file of fileList) {
    const slot = classify(file);
    if (!slot) continue;
    body.append(slot, file, file.name);
    count += 1;
  }
  if (!count) {
    note("این پسوند پذیرفته نمی‌شود. NIfTI یا PNG بفرست.");
    return;
  }
  const response = await fetch("/api/sessions", { method: "POST", body });
  const payload = await response.json();
  if (!response.ok) {
    note(payload.detail || "بارگذاری ناموفق بود.");
    return;
  }
  applySession(payload);
}

async function loadSample() {
  const response = await fetch("/api/sessions/sample", { method: "POST" });
  applySession(await response.json());
  note("این حجم مصنوعی است و MRI یک بیمار نیست.");
}

function applySession(payload) {
  state.id = payload.id;
  state.modalities = payload.modalities;
  state.image = payload.image;
  state.shape = payload.shape;
  state.space = "original";
  state.overlay = 0;
  state.modality = payload.modalities.includes("flair") ? "flair" : (payload.modalities[0] || "flair");
  renderSlots(payload.modalities);
  renderModalities();
  resetIndex();
  document.querySelector("#predict").disabled = false;
  document.querySelector("#empty").hidden = true;
  showSlice();
  setVerdict("waiting", "آماده بررسی", payload.image && !payload.modalities.length
    ? "این یک عکس دوبعدی است. مدل چهار حجم سه‌بعدی می‌خواهد."
    : "بررسی را بزن. اگر مدل آموزش ندیده باشد، پاسخی ساخته نمی‌شود.");
}

function renderModalities() {
  const host = document.querySelector("#modalities");
  host.innerHTML = state.modalities.map((name) => `<button type="button" data-modality="${name}">${name}</button>`).join("");
  host.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("on", button.dataset.modality === state.modality);
    button.addEventListener("click", () => {
      state.modality = button.dataset.modality;
      renderModalities();
      showSlice();
    });
  });
}

function resetIndex() {
  const axis = axisIndex[state.axis];
  const max = state.shape ? Math.max(state.shape[axis] - 1, 0) : 0;
  const slider = document.querySelector("#index");
  slider.max = String(max);
  slider.disabled = !state.shape;
  state.index = Math.floor(max / 2);
  slider.value = String(state.index);
}

function showSlice() {
  if (!state.id) return;
  const image = document.querySelector("#slice");
  if (state.image && !state.shape) {
    image.src = `/api/sessions/${state.id}/slice?t=${Date.now()}`;
    image.classList.add("show");
    document.querySelector("#slice-label").textContent = "عکس";
    return;
  }
  const params = new URLSearchParams({
    modality: state.modality,
    axis: state.axis,
    index: String(state.index),
    overlay: String(state.overlay),
    space: state.space,
  });
  image.src = `/api/sessions/${state.id}/slice?${params}`;
  image.classList.add("show");
  document.querySelector("#slice-label").textContent = `برش ${state.index}`;
}

async function predict() {
  if (!state.id) return;
  document.querySelector("#predict").disabled = true;
  setVerdict("waiting", "در حال بررسی", "اگر حجم بزرگ باشد این مرحله طول می‌کشد.");
  const response = await fetch(`/api/sessions/${state.id}/predict`, { method: "POST" });
  const payload = await response.json();
  document.querySelector("#predict").disabled = false;
  if (!response.ok) {
    setVerdict("waiting", "بررسی متوقف شد", payload.detail || "خطای سرور");
    return;
  }
  if (!payload.available) {
    setVerdict("waiting", "پاسخی داده نشد", payload.message);
    document.querySelector("#result-note").textContent = payload.disclaimer || "";
    return;
  }
  state.space = payload.space || "aligned";
  state.shape = payload.shape || state.shape;
  state.overlay = payload.marked ? 1 : 0;
  resetIndex();
  showSlice();
  setVerdict(payload.marked ? "marked" : "clear", payload.marked ? "ناحیه مشکوک علامت خورد" : "ناحیه‌ای علامت نخورد", payload.message);
  const regions = document.querySelector("#regions");
  const entries = Object.entries(payload.regions || {});
  regions.hidden = entries.length === 0;
  regions.innerHTML = entries.map(([name, info]) => `<div><dt>${name}</dt><dd>${info.voxels.toLocaleString("fa-IR")} وکسل</dd></div>`).join("");
  document.querySelector("#result-note").textContent = payload.disclaimer;
}

function setVerdict(kind, title, copy) {
  const node = document.querySelector("#verdict");
  node.className = `verdict ${kind}`;
  node.innerHTML = `<b>${title}</b><span>${copy}</span>`;
}

function note(text) {
  document.querySelector("#intake-note").textContent = text;
}
