// Called after every step: which paintings are fully in view, roughly head-on, from the segment right in
// front of their gate (so no other gate occludes them) and within 2-10 blocks.  Environment-side ground truth
// for choosing fair questions; never shown to the agent.
() => {
  const S = window.__mcgallery; if (!S) return null;
  const cam = game.camera; cam.updateMatrixWorld();
  const pz = player.position.z;
  let yaw = ((player.yaw % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
  const now = [];
  if (Math.abs(yaw) < 0.45 && Math.abs(player.pitch) < 0.5) {
    for (let i = 0; i < S.meshes.length; i++) {
      const m = S.meshes[i]; const zg = S.gates[m.gate];
      const prevZ = m.gate === 0 ? S.z0 + 3 : S.gates[m.gate - 1];
      if (!(pz > zg + 1 && pz < prevZ)) continue;
      const dist = pz - m.cz; if (dist > 10 || dist < 2) continue;
      let ok = true;
      for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        const v = new THREE.Vector3(m.cx + dx, m.cy + dy, m.cz).project(cam);
        if (Math.abs(v.x) > 0.97 || Math.abs(v.y) > 0.97 || v.z > 1) { ok = false; break; }
      }
      if (ok) { S.seen[i] = (S.seen[i] || 0) + 1; now.push(i); }
    }
  }
  if (Math.abs(yaw) < 1e9) window.mc.setDay();
  return { seen: Object.keys(S.seen).map(Number), now, pos: [player.position.x, player.position.y, player.position.z],
           yaw, passed: S.gates.filter(z => pz < z).length, atEnd: pz < S.gates[S.gates.length - 1] - 1.5 };
}
