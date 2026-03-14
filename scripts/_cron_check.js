const { execSync } = require("child_process");

const jobs = [
  ["morning_brief",      "39e1e9ba-9ed0-4e50-b577-51c235946ef1", "07:00"],
  ["pre_market_check",   "8a914e56-708d-418a-b4d2-7059568c4e0b", "08:00"],
  ["midday_check",       "2365e0be-2ba3-4836-b81a-b9f306319243", "12:00"],
  ["post_market_analysis","91382f28-6c48-43e5-a4cc-c686e95844d5", "16:00"],
  ["mining_start_check", "6a8ff852-06f1-4497-bde5-934dbbd953e9", "18:15"],
  ["mining_review",      "8fd7cb3c-634e-48e2-b711-396ebd21c6df", "20:00"],
  ["project_improvement","c2572226-0967-4d0d-8b9d-84e2989bfa68", "22:00"],
  ["overnight_check",    "cee80b37-8c70-4f79-bdef-c7b3b43b7f18", "02:00"],
];

for (const [name, id, sched] of jobs) {
  try {
    const raw = execSync(`openclaw cron runs --id ${id} --limit 3`, { encoding: "utf8", stdio: ["pipe","pipe","pipe"] });
    const lines = raw.split("\n");
    let jsonStart = -1;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim().startsWith("{")) { jsonStart = i; break; }
    }
    if (jsonStart < 0) {
      console.log(`${name} (${sched}): no json output`);
      continue;
    }
    const d = JSON.parse(lines.slice(jsonStart).join("\n"));
    console.log(`${name} (${sched}) - ${d.total} runs:`);
    if (d.entries.length === 0) {
      console.log("  (no runs yet)");
      continue;
    }
    for (const e of d.entries) {
      const t = new Date(e.runAtMs).toLocaleString("ko-KR", {
        timeZone: "Asia/Seoul",
        month: "2-digit", day: "2-digit",
        hour: "2-digit", minute: "2-digit",
        hour12: false,
      });
      const dur = Math.round(e.durationMs / 1000);
      console.log(`  ${t} | ${e.status.padEnd(5)} | ${String(dur).padStart(3)}s | dlv=${e.deliveryStatus}`);
    }
  } catch (err) {
    console.log(`${name} (${sched}): error - ${err.message.split("\n")[0]}`);
  }
}
