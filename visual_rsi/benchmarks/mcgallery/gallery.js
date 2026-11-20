// Injected into the GameWorld Minecraft Clone page (18_minecraft-clone-glm) by mcgallery_backend.py.
// Builds a straight walled corridor along -Z from the player, with numbered cross-walls ("gates") that
// each have a 1-wide doorway and two framed 2x2-block paintings facing the approaching player.
// Returns the geometry the backend needs for scoring.  No game file is modified.
(args) => {
  const { paintings, nGates, gap, firstGap } = args;   // paintings: [{gate, side, png}] data-URL textures
  const B = BlockType;
  const p = player.position;
  const x0 = Math.floor(p.x), z0 = Math.floor(p.z), yf = Math.floor(p.y + 0.01) - 1;
  const zEnd = z0 - firstGap - gap * (nGates - 1) - 6;
  const set = (x, y, z, t) => chunkManager.setBlock(x, y, z, t);
  for (let z = zEnd; z <= z0 + 3; z++) {
    for (let x = x0 - 4; x <= x0 + 4; x++) {
      set(x, yf, z, B.STONE);
      for (let y = yf + 1; y <= yf + 6; y++) set(x, y, z, (Math.abs(x - x0) === 3 && y <= yf + 4) ? B.COBBLESTONE : B.AIR);
    }
  }
  for (let x = x0 - 3; x <= x0 + 3; x++) for (let y = yf + 1; y <= yf + 4; y++) set(x, y, z0 + 3, B.COBBLESTONE);
  for (let x = x0 - 3; x <= x0 + 3; x++) for (let y = yf + 1; y <= yf + 4; y++) set(x, y, zEnd, B.GOLD_ORE);
  const gates = [];
  for (let k = 0; k < nGates; k++) {
    const zg = z0 - firstGap - gap * k;
    for (let x = x0 - 2; x <= x0 + 2; x++) for (let y = yf + 1; y <= yf + 4; y++) {
      const door = (x === x0 && y <= yf + 2);
      set(x, y, zg, door ? B.AIR : B.WOOD_PLANK);
    }
    gates.push(zg);
  }
  for (const c of chunkManager.chunks.values()) if (c.needsUpdate && c.isGenerated) chunkManager.buildChunkMesh(c);
  // paintings and gate numbers: unlit planes slightly in front of the wall face that faces +Z
  const group = new THREE.Group(); group.name = 'mcgallery';
  const loader = (url) => { const img = new Image(); img.src = url; const t = new THREE.Texture(img);
    img.onload = () => { t.needsUpdate = true; }; t.magFilter = THREE.NearestFilter; t.minFilter = THREE.NearestFilter; return t; };
  const meshes = [];
  for (const pt of paintings) {
    const zg = gates[pt.gate];
    const cx = x0 + 0.5 + (pt.side === 'L' ? -1.5 : 1.5);
    const frame = new THREE.Mesh(new THREE.PlaneGeometry(2.15, 2.15), new THREE.MeshBasicMaterial({ color: 0x3b2a1a }));
    frame.position.set(cx, yf + 1 + 1.55, zg + 1 + 0.01);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: loader(pt.png), fog: false }));
    m.position.set(cx, yf + 1 + 1.55, zg + 1 + 0.02);
    group.add(frame); group.add(m);
    meshes.push({ gate: pt.gate, side: pt.side, cx, cy: yf + 2.55, cz: zg + 1.02 });
  }
  for (let k = 0; k < nGates; k++) {
    const cv = document.createElement('canvas'); cv.width = cv.height = 64;
    const g = cv.getContext('2d'); g.fillStyle = '#f2e6c8'; g.fillRect(0, 0, 64, 64);
    g.fillStyle = '#1a1a1a'; g.font = 'bold 52px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(k + 1), 32, 35);
    const t = new THREE.CanvasTexture(cv); t.magFilter = THREE.NearestFilter;
    const s = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.9), new THREE.MeshBasicMaterial({ map: t, fog: false }));
    s.position.set(x0 + 0.5, yf + 3.5, gates[k] + 1 + 0.02); group.add(s);
  }
  game.scene.add(group);
  window.__mcgallery = { x0, z0, yf, gates, zEnd, meshes, seen: {} };
  // stand in the corridor centre facing -Z, daytime, no mobs
  p.set(x0 + 0.5, yf + 1, z0 + 1.5); player.velocity.set(0, 0, 0); player.yaw = 0; player.pitch = 0;
  if (typeof mobManager !== 'undefined' && mobManager) { for (const mob of mobManager.mobs) game.scene.remove(mob.mesh); mobManager.mobs = []; }
  window.mc.setDay();
  return { x0, z0, yf, gates, zEnd };
}
