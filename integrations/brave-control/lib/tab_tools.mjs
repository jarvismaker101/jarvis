/**
 * Helper for formatting brave-control tab lists.
 * Pure function -> easy to unit test with fake page objects.
 * Each tabInfo: { index: number, title: string, url: string, active: boolean }
 */
export function formatTabs(tabInfos) {
  if (!Array.isArray(tabInfos) || tabInfos.length === 0) {
    return "No tabs open.";
  }
  const lines = tabInfos.map((t) => {
    const idx = t.index ?? 0;
    const title = (t.title ?? "").trim() || "(no title)";
    const url = (t.url ?? "").trim() || "(no url)";
    const marker = t.active ? " [active]" : "";
    return `${idx}: "${title}" - ${url}${marker}`;
  });
  return lines.join("\n");
}

/**
 * Build tabInfos from real Playwright page objects.
 * Handles both sync stubs and async title() promises.
 * activePage is the currently targeted page (page variable in server.mjs).
 */
export async function buildTabInfos(pages, activePage) {
  const infos = await Promise.all(
    pages.map(async (pg, idx) => {
      let url = "";
      let title = "";
      try {
        url = typeof pg.url === "function" ? pg.url() : (pg.url ?? "");
      } catch {}
      try {
        const raw = typeof pg.title === "function" ? pg.title() : (pg.title ?? "");
        // real playwright title() returns Promise; stub may return string
        title = raw instanceof Promise ? await raw : raw;
        title = title ?? "";
      } catch {}
      return {
        index: idx,
        title: String(title),
        url: String(url),
        active: pg === activePage,
      };
    })
  );
  return infos;
}
