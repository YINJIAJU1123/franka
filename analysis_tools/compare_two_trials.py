"""Compare two completed ACT runs without live connections or runtime changes."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/lixin/act_gripper_temporal_run02_20261112')
OUT = ROOT / 'reports/retest2_comparison_20261112'
sys.path.insert(0, str(ROOT))
from franka_gripper_control import CartesianServo


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def metric(x):
    return {'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)), 'max': float(np.max(x))}


def vector(text):
    return np.fromstring(text.strip('[]'), sep=' ')


def edges(binary):
    return np.flatnonzero(np.diff(np.r_[0, binary]) != 0)


def arc_sample(x, count=500):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
    valid = np.r_[True, np.diff(arc) > 1e-9]
    if arc[-1] < 1e-8:
        return np.repeat(x[:1], count, axis=0)
    return np.stack([np.interp(np.linspace(0, arc[-1], count), arc[valid], x[valid, j]) for j in range(3)], axis=1)


def main():
    OUT.mkdir(exist_ok=True)
    original_socket = socket.socket
    def deny(family=socket.AF_INET, *a, **kw):
        if family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError('Live IP sockets forbidden during trial comparison')
        return original_socket(family, *a, **kw)
    socket.socket = deny
    servo = CartesianServo(str(ROOT / 'fr3.urdf'))
    runs = {}
    files = {'first_success': ('real_20261112_164844.csv', 'onsite_temporal_20261112_164839.log', 'onsite_temporal_20261112_164839.exit'),
             'retest2': ('real_20261112_170932.csv', 'onsite_temporal_retest_20261112_170928.log', 'onsite_temporal_retest_20261112_170928.exit')}
    arrays = {}
    for name, (csvname, logname, exitname) in files.items():
        path = ROOT / 'reports/real_temporal' / csvname
        rows = list(csv.DictReader(path.open()))
        absolute = np.array([dt.datetime.fromisoformat(x['wall_time']).timestamp() for x in rows])
        time = absolute - absolute[0]
        q = np.stack([vector(x['q_feedback']) for x in rows])
        target = np.stack([vector(x['q_target']) for x in rows])
        poses = [servo.fk(x) for x in q]
        xyz = np.stack([x[0] for x in poses]); quat = np.stack([x[1] for x in poses])
        command = np.array([int(float(x['gripper_cmd'])) for x in rows])
        feedback = np.array([int(float(x['gripper_feedback'])) for x in rows])
        raw = np.array([float(x['gripper_action_raw']) for x in rows])
        width = np.array([float(x['gripper_width']) for x in rows])
        event_rows = []
        for i in edges(command):
            feedback_after = np.flatnonzero(feedback[i:] == command[i])
            event_rows.append({'type': 'close' if command[i] else 'open', 'index': int(i), 'time_s': float(time[i]),
                               'wall_time': rows[i]['wall_time'],
                               'width_feedback_delay_s': float(time[i + feedback_after[0]] - time[i]) if len(feedback_after) else None,
                               'xyz_m': xyz[i].tolist(), 'quaternion_xyzw': quat[i].tolist()})
        one_cycle = [x['type'] for x in event_rows] == ['close', 'open']
        assert one_cycle, event_rows
        close, opened = [x['index'] for x in event_rows]
        terminal = xyz[time >= time[-1] - 2].mean(0)
        distance = np.linalg.norm(xyz - terminal, axis=1)
        all_remaining_within = np.logical_and.accumulate((distance <= .002)[::-1])[::-1]
        stable_candidates = np.flatnonzero(all_remaining_within & (np.arange(len(time)) >= opened) & (time <= time[-1] - 2))
        assert len(stable_candidates), 'No two-second terminal stable interval'
        completed = int(stable_candidates[0])
        segments = {'approach': (0, close), 'closed_carry': (close, opened), 'post_release_lift': (opened, completed)}
        error = np.stack([vector(x['position_error']) for x in rows])
        episode = np.array([int(x['episode']) for x in rows]); steps = np.array([int(x['step']) for x in rows])
        sequence = np.array([int(x['state_seq']) for x in rows])
        log = (ROOT / 'reports' / logname).read_text(errors='replace')
        summary = {'csv': str(path), 'csv_sha256': sha(path), 'rows': len(rows), 'wall_time_start': rows[0]['wall_time'],
                   'wall_time_last': rows[-1]['wall_time'], 'record_duration_s': float(time[-1]),
                   'average_policy_hz': float((len(time)-1)/time[-1]), 'loop_interval_s': metric(np.diff(time)),
                   'steps_contiguous': bool(np.array_equal(steps, np.arange(len(rows)))), 'episodes': np.unique(episode).tolist(),
                   'repeated_state_sequence_fraction': float(np.mean(np.diff(sequence)==0)),
                   'state_sequence_backwards_count': int(np.sum(np.diff(sequence)<0)),
                   'one_close_one_open': one_cycle, 'command_events': event_rows,
                   'feedback_event_count': len(edges(feedback)), 'width_range_mm': [float(width.min()*1000), float(width.max()*1000)],
                   'held_width_median_mm': float(np.median(width[(time>time[close]+2)&(time<time[opened]-1)])*1000),
                   'start_xyz_m': xyz[0].tolist(), 'terminal_xyz_mean_m': terminal.tolist(),
                   'terminal_joint_mean_rad': q[time>=time[-1]-2].mean(0).tolist(),
                   'completion_proxy_definition': 'After open, first sample within 2mm of terminal mean for all remaining samples, with at least 2 seconds observed afterwards.',
                   'completion_proxy_s': float(time[completed]), 'post_completion_recorded_s': float(time[-1]-time[completed]),
                   'phase_durations_s': {k: float(time[h]-time[l]) for k,(l,h) in segments.items()},
                   'phase_path_lengths_mm': {k: float(np.linalg.norm(np.diff(xyz[l:h+1],axis=0),axis=1).sum()*1000) for k,(l,h) in segments.items()},
                   'tracking_position_error_mm': metric(np.linalg.norm(error[:completed+1],axis=1)*1000),
                   'approach_descent_until_close_mm': float((xyz[0,2]-xyz[close,2])*1000),
                   'lift_after_open_to_terminal_mm': float((terminal[2]-xyz[opened,2])*1000),
                   'exit_code': int((ROOT/'reports'/exitname).read_text()), 'onsite_C_observed': '>>> [C] exit' in log,
                   'exception_in_ACT_console': 'Traceback (most recent call last)' in log,
                   'EMA_step30000_load_logged': 'step=30000' in log,
                   'temporal_001_banner_logged': 'native temporal k=0.01' in log}
        runs[name] = {'summary': summary, 'time': time, 'xyz': xyz, 'quat': quat, 'q': q, 'raw': raw,
                      'command': command, 'feedback': feedback, 'width': width, 'segments': segments, 'completed': completed}
        for key in ['time','xyz','quat','q','raw','command','feedback','width']:
            arrays[name+'_'+key] = runs[name][key]
    a, b = runs['first_success'], runs['retest2']
    def pair(index_a, index_b):
        delta = b['xyz'][index_b]-a['xyz'][index_a]
        angle = (Rotation.from_quat(b['quat'][index_b])*Rotation.from_quat(a['quat'][index_a]).inv()).magnitude()
        return {'xyz_delta_mm': (delta*1000).tolist(), 'distance_mm': float(np.linalg.norm(delta)*1000),
                'orientation_difference_deg': float(np.rad2deg(angle)),
                'max_joint_difference_rad': float(np.abs(b['q'][index_b]-a['q'][index_a]).max())}
    comparison = {'start': pair(0,0), 'close': pair(a['segments']['approach'][1],b['segments']['approach'][1]),
                  'open': pair(a['segments']['closed_carry'][1],b['segments']['closed_carry'][1]),
                  'end': pair(len(a['time'])-1,len(b['time'])-1), 'phases': {}}
    for phase in a['segments']:
        la, ha = a['segments'][phase]; lb, hb = b['segments'][phase]
        xa, xb = arc_sample(a['xyz'][la:ha+1]), arc_sample(b['xyz'][lb:hb+1])
        aligned = np.linalg.norm(xa-xb,axis=1)*1000
        nearest = np.r_[cKDTree(xa).query(xb)[0],cKDTree(xb).query(xa)[0]]*1000
        comparison['phases'][phase] = {'arc_normalized_point_distance_mm': metric(aligned),
                                       'symmetric_nearest_curve_distance_mm': metric(nearest),
                                       'duration_delta_s': b['summary']['phase_durations_s'][phase]-a['summary']['phase_durations_s'][phase]}
        arrays[phase+'_arc_aligned_distance_mm'] = aligned
    comparison['close_time_delta_s'] = b['summary']['command_events'][0]['time_s']-a['summary']['command_events'][0]['time_s']
    comparison['open_time_delta_s'] = b['summary']['command_events'][1]['time_s']-a['summary']['command_events'][1]['time_s']
    comparison['completion_proxy_delta_s'] = b['summary']['completion_proxy_s']-a['summary']['completion_proxy_s']
    integrity = json.loads((ROOT/'source_integrity.json').read_text())
    result = {'frame': 'Same frozen FR3 URDF, fr3_link8; this is flange pose, not measured block position.',
              'model': 'run02 EMA step30000', 'temporal_ensemble_coeff': .01,
              'user_confirmed_both_tasks_completed': True,
              'alignment': 'Gripper phase anchors + normalized arc length; real-time timing differences reported separately, no DTW.',
              'runs': {k:v['summary'] for k,v in runs.items()}, 'comparison': comparison,
              'limitations': ['No runtime images saved; actual block landing coordinates cannot be measured from these logs.',
                             'Two successes in the same setup do not establish general robustness.',
                             'Completion is a defined position-settling proxy, not a vision-based task-completion timestamp.'],
              'existing_source_files_unchanged': all(sha(p)==h for p,h in integrity.items()),
              'robot_commands_sent_by_analysis': 0}
    assert result['existing_source_files_unchanged']
    (OUT/'comparison_results.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'comparison_traces.npz',**arrays)

    fig=plt.figure(figsize=(16,8.5),layout='constrained')
    axes=[fig.add_subplot(2,3,1,projection='3d')]+[fig.add_subplot(2,3,i) for i in range(2,7)]
    colors=['#207d8d','#cc6842'];labels=['First success','RETEST 2']
    for (name,run),color,label in zip(runs.items(),colors,labels):
        end=run['completed'];t=run['time'];x=run['xyz']*1000
        axes[0].plot(x[:end+1,0],x[:end+1,1],x[:end+1,2],color=color,label=label,lw=1.2)
        for event,marker in zip(run['summary']['command_events'],['o','x']):
            i=event['index'];axes[0].scatter(x[i,0],x[i,1],x[i,2],color=color,marker=marker,s=30)
        axes[1].plot(t[:end+1],x[:end+1,2],color=color,label=label)
        axes[2].plot(t,run['raw'],color=color,label=label)
        axes[3].plot(t,run['width']*1000,color=color,label=label)
    axes[0].set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title='Actual flange paths: close (o), open (x)');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Elapsed policy time (s)',ylabel='Actual Z (mm)',title='Physical progression, tail hold excluded');axes[1].legend()
    axes[2].axhline(.55,ls=':',color='grey');axes[2].axhline(.45,ls=':',color='grey');axes[2].set(xlabel='Elapsed policy time (s)',ylabel='Raw gripper action',title='Gripper commands: timing differs');axes[2].legend()
    axes[3].set(xlabel='Elapsed policy time (s)',ylabel='Measured width (mm)',title='Width feedback: one close and one open');axes[3].legend()
    phase_labels=['Approach','Carry','Post-release lift'];phase_colors=['#3b8c6e','#8d72aa','#b5883e']
    for phase,label,color in zip(a['segments'],phase_labels,phase_colors):axes[4].plot(np.linspace(0,1,500),arrays[phase+'_arc_aligned_distance_mm'],color=color,label=label)
    axes[4].set(xlabel='Normalized distance along each phase',ylabel='Path separation (mm)',title='Geometry comparison independent of speed');axes[4].legend(fontsize=8)
    left=np.zeros(2)
    for phase,label,color in zip(a['segments'],phase_labels,phase_colors):
        values=[run['summary']['phase_durations_s'][phase] for run in runs.values()];axes[5].barh([0,1],values,left=left,color=color,label=label);left+=values
    axes[5].set(yticks=[0,1],yticklabels=labels,xlabel='Elapsed time to 2mm terminal-settling proxy (s)',title='Phase duration; manual tail hold excluded');axes[5].legend(fontsize=7)
    for ax in axes[1:]:ax.grid(alpha=.18)
    fig.suptitle('Two onsite-confirmed successful trials | Same run02 EMA step30,000 + temporal k=0.01',fontsize=14)
    fig.savefig(OUT/'two_successful_trials.png',dpi=160)
    print('COMPARISON_DONE',OUT,flush=True)
    print('SUMMARY',json.dumps({'runs':result['runs'],'comparison':comparison}),flush=True)


if __name__=='__main__':
    main()
