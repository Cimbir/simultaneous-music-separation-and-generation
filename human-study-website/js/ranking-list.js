export function createRankingList(listEl, clips, { labelFor }) {
  let order = clips.map((clip) => clip.clip_id);
  let draggedId = null;

  function swap(i, j) {
    if (j < 0 || j >= order.length) return;
    [order[i], order[j]] = [order[j], order[i]];
    render();
  }

  function moveBefore(movedId, targetId) {
    const from = order.indexOf(movedId);
    order.splice(from, 1);
    order.splice(order.indexOf(targetId), 0, movedId);
    render();
  }

  function render() {
    listEl.innerHTML = "";
    order.forEach((clipId, position) => {
      const clip = clips.find((c) => c.clip_id === clipId);
      const item = renderItem(position + 1, labelFor(clip), order.length);
      item.dataset.clip = clipId;

      item.querySelector(".up").addEventListener("click", () => swap(position, position - 1));
      item.querySelector(".down").addEventListener("click", () => swap(position, position + 1));
      enableDragAndDrop(item);
      listEl.appendChild(item);
    });
  }

  function enableDragAndDrop(item) {
    item.addEventListener("dragstart", () => {
      draggedId = item.dataset.clip;
      item.classList.add("dragging");
    });
    item.addEventListener("dragend", () => {
      draggedId = null;
      item.classList.remove("dragging");
    });
    item.addEventListener("dragover", (event) => {
      event.preventDefault();
      item.classList.add("over");
    });
    item.addEventListener("dragleave", () => item.classList.remove("over"));
    item.addEventListener("drop", (event) => {
      event.preventDefault();
      item.classList.remove("over");
      if (draggedId && draggedId !== item.dataset.clip) {
        moveBefore(draggedId, item.dataset.clip);
      }
    });
  }

  render();
  return { order: () => [...order] };
}

function renderItem(rank, label, total) {
  const li = document.createElement("li");
  li.className = "rankitem";
  li.draggable = true;
  li.innerHTML = `
    <span class="rank-num">${rank}</span>
    <span class="rank-name">Clip ${label}</span>
    <span class="arrows">
      <button class="up" aria-label="Move up" ${rank === 1 ? "disabled" : ""}>▲</button>
      <button class="down" aria-label="Move down" ${rank === total ? "disabled" : ""}>▼</button>
    </span>
    <span class="grip">⋮⋮</span>`;
  return li;
}
