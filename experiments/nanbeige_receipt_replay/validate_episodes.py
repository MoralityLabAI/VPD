"""Deterministically replay saved structured episodes without another model call."""
import argparse
import json
from pathlib import Path
from .ledger_eval import Case,run_episode
from .io import digest,save


def validate(record):
    events=iter(record['events']);calls=0
    def replay(**kwargs):
        nonlocal calls
        event=next(events);calls+=1
        assert kwargs['messages']==event['messages'],'History mismatch'
        assert kwargs['seed']==event['sample_seed'],'Seed mismatch'
        assert kwargs['metadata']==event['metadata'],'Out-of-band metadata mismatch'
        if 'response' not in event:raise RuntimeError('Recorded backend failure')
        return event['response']
    rebuilt=run_episode(Case(**record['case']),replay,record['settings'])
    for key in ['initial_state','final_state','labels','complete','rating']:
        assert rebuilt[key]==record[key],key
    assert len(rebuilt['events'])==len(record['events'])==calls
    for old,new in zip(record['events'],rebuilt['events']):
        for key in ['before','after','valid','parsed','chosen_action','native_reasoning','visible_answer']:
            assert old.get(key)==new.get(key),key
    return calls


def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in sorted(a.episodes.glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'));turns=validate(record)
        rows.append(dict(episode_id=record['episode_id'],sha256=digest(path),turns=turns))
    save(a.output,dict(status='saved_episodes_replayed',episodes=len(rows),turns=sum(r['turns'] for r in rows),records=rows,
        scope='Exact histories, seeds, deterministic environment transitions and labels verified; no additional inference'))


if __name__=='__main__':main()
