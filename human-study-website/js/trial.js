import { createPlayerDeck } from "./player-deck.js";
import { createRankingList } from "./ranking-list.js";

const CLIP_LABELS = ["A", "B", "C", "D"];

export function createTrial(refs, clips, { maxPlays, onReadyChange }) {
  const startedAt = new Date().toISOString();
  const labelFor = (clip) => CLIP_LABELS[clips.indexOf(clip)];

  const deck = createPlayerDeck(refs.players, clips, {
    maxPlays,
    labelFor,
    onChange: () => onReadyChange(isReady()),
  });

  const musicalityRanking = createRankingList(refs.musicalityList, clips, { labelFor });
  const coherenceRanking = createRankingList(refs.coherenceList, clips, { labelFor });

  function isReady() {
    return deck.everyClipPlayed();
  }

  return {
    isReady,
    stopAudio: deck.stop,
    answers() {
      return {
        clips: clips.map((clip, position) => ({
          clip_id: clip.clip_id, model: clip.model, src: clip.src, position,
        })),
        musicality: musicalityRanking.order(),
        coherence: coherenceRanking.order(),
        replays: deck.replayCounts(),
        startedAt,
      };
    },
  };
}
