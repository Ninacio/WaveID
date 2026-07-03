import type { QueryMatch } from "@/lib/api"

export function buildCasualExplanation(
  match: QueryMatch,
  rank: number,
  similarityGap: number
): { headline: string; body: string; caution?: string } {
  const simPct = (match.similarity * 100).toFixed(0)
  const covPct = (match.coverage * 100).toFixed(0)

  if (match.match_strength === "strong") {
    return {
      headline: "This is very likely the same recording",
      body: `About ${covPct}% of your clip lines up with this track, and the audio fingerprints are a ${simPct}% match. Listen to both — they should sound like the same song.`,
    }
  }

  if (match.match_strength === "weak" && rank === 0 && similarityGap < 0.04) {
    return {
      headline: "Similar sound, but we're not sure it's the same song",
      body: `Your clip shares some acoustic qualities with this track (${simPct}% fingerprint overlap across ${covPct}% of the clip). That can happen when songs share a genre, tempo, or production style — even if they're different tracks.`,
      caution:
        "Compare the two players above. If the melody or vocals clearly differ, treat this as a false alarm — WaveID matched the vibe, not necessarily the song.",
    }
  }

  if (match.match_strength === "weak") {
    return {
      headline: "Probably not the same song",
      body: `Only a partial overlap was detected (${simPct}% similarity). This reference track is unlikely to be what you're hearing.`,
    }
  }

  return {
    headline: "Some similarities detected — listen to compare",
    body: `Roughly ${covPct}% of your clip has moments that resemble this track (${simPct}% fingerprint match). Use the players to check whether the melody and vocals actually match.`,
    caution:
      rank > 0
        ? "Another catalogue track scored almost as highly. When two results are this close, trust your ears over the percentage."
        : undefined,
  }
}
