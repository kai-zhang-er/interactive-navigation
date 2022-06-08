from __future__ import print_function
import math
import random
import time
import numpy as np

from matplotlib.pyplot import ylim

from pddlstream.algorithms.meta import solve, create_parser
from utils.pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics
from utils.pybullet_tools.pr2_primitives import APPROACH_DISTANCE, GRASP_LENGTH, Command, Conf, Grasp, Pose, get_ik_fn, get_ik_ir_gen, get_motion_gen, \
    get_stable_gen, get_grasp_gen, Attach, Detach, Clean, Cook, control_commands, \
    get_gripper_joints, GripperCommand, apply_commands, State
from utils.pybullet_tools.pr2_problems import Problem, cleaning_problem, cooking_problem, create_pr2
from utils.pybullet_tools.pr2_utils import GET_GRASPS, REST_LEFT_ARM, TOP_HOLDING_LEFT_ARM, arm_conf, close_arm, get_arm_joints, ARM_NAMES, get_base_pose, get_carry_conf, get_gripper_link, get_group_joints, get_group_conf, get_other_arm, get_top_grasps, load_inverse_reachability, open_arm, set_arm_conf, set_group_conf
from utils.pybullet_tools.utils import CIRCULAR_LIMITS, BodySaver, NewPose, connect, create_box, get_link_pose, get_pose, get_unit_vector, invert, is_placement, multiply, point_from_pose, \
    disconnect, get_joint_positions, enable_gravity, save_state, restore_state, HideOutput, \
    get_distance, LockRenderer, get_max_limit, has_gui, WorldSaver, set_camera_pose, set_point, set_pose, uniform_pose_generator, unit_quat, wait_if_gui, add_line, SEPARATOR
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, fn_from_constant, empty_gen, from_test
from pddlstream.language.constants import Equal, AND, print_solution, PDDLProblem
from pddlstream.utils import elapsed_time, read, INF, get_file_path, find_unique, Profiler, str_from_object
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.language.object import SharedOptValue
from pddlstream.language.external import defer_shared, never_defer
from examples.pybullet.tamp.streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    move_cost_fn
from collections import namedtuple

BASE_CONSTANT = 1
BASE_VELOCITY = 0.5

def place_movable(certified):
    for literal in certified:
        if literal[0] != 'not':
            continue
        fact = literal[1]
        if fact[0] == 'trajposecollision':
            _, b, p = fact[1:]
            p.assign()
        if fact[0] == 'trajarmcollision':
            _, a, q = fact[1:]
            q.assign()
        if fact[0] == 'trajgraspcollision':
            _, a, o, g = fact[1:]
            # TODO: finish this

# def get_base_motion_synth(problem, teleport=False):
#     # TODO: could factor the safety checks if desired (but no real point)
#     #fixed = get_fixed(robot, movable)
#     def fn(outputs, certified):
#         assert(len(outputs) == 1)
#         q0, _, q1 = find_unique(lambda f: f[0] == 'basemotion', certified)[1:]
#         place_movable(certified)
#         free_motion_fn = get_motion_gen(problem, teleport)
#         return free_motion_fn(q0, q1)
#     return fn

def move_cost_fn(c):
    [t] = c.commands
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    #return BASE_CONSTANT + distance / BASE_VELOCITY
    return 1

#######################################################

def extract_point2d(v):
    if isinstance(v, Conf):
        return v.values[:2]
    if isinstance(v, Pose):
        return point_from_pose(v.value)[:2]
    if isinstance(v, SharedOptValue):
        if v.stream == 'sample-pose':
            r, = v.values
            return point_from_pose(get_pose(r))[:2]
        if v.stream == 'inverse-kinematics':
            p, = v.values
            return extract_point2d(p)
    if isinstance(v, CustomValue):
        if v.stream == 'p-sp':
            r, = v.values
            return point_from_pose(get_pose(r))[:2]
        if v.stream == 'q-ik':
            p, = v.values
            return extract_point2d(p)
    raise ValueError(v.stream)

def opt_move_cost_fn(t):
    # q1, q2 = t.values
    # distance = get_distance(extract_point2d(q1), extract_point2d(q2))
    #return BASE_CONSTANT + distance / BASE_VELOCITY
    return 1

#######################################################

CustomValue = namedtuple('CustomValue', ['stream', 'values'])

def opt_pose_fn(o, r):
    p = CustomValue('p-sp', (r,))
    return p,

def opt_ik_fn(a, o, p, g):
    q = CustomValue('q-ik', (p,))
    t = CustomValue('t-ik', tuple())
    return q, t

def opt_motion_fn(q1, q2):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

#######################################################

def pddlstream_from_problem(problem, collisions=True, teleport=False):
    robot = problem.robot

    # TODO: integrate pr2 & tamp
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    #initial_bq = Pose(robot, get_pose(robot))
    initial_bq = Conf(robot, get_group_joints(robot, 'base'), get_group_conf(robot, 'base'))
    init = [
        ('CanMove',),
        ('BConf', initial_bq),
        ('AtBConf', initial_bq),
        Equal(('PickCost',), 1),
        Equal(('PlaceCost',), 1),
    ] + [('Sink', s) for s in problem.sinks] + \
           [('Stove', s) for s in problem.stoves] + \
           [('Connected', b, d) for b, d in problem.buttons] + \
           [('Button', b) for b, _ in problem.buttons]
    for arm in ARM_NAMES:
    #for arm in problem.arms:
        joints = get_arm_joints(robot, arm)
        conf = Conf(robot, joints, get_joint_positions(robot, joints))
        init += [('Arm', arm), ('AConf', arm, conf), ('HandEmpty', arm), ('AtAConf', arm, conf)]
        if arm in problem.arms:
            init += [('Controllable', arm)]
    for body in problem.movable:
        pose = Pose(body, get_pose(body))
        init += [('Graspable', body), ('Pose', body, pose),
                 ('AtPose', body, pose)]
        for surface in problem.surfaces:
            init += [('Stackable', body, surface)]
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    goal = [AND]
   
    # goal += [('AtBConf', initial_bq)]

    # body goal position
    # pose = Pose(11, get_pose(body))
    # pose_value=[(-2.5,-1,0.),(0.0,0.0,0.0,1.0)]
    # pose=Pose(12, pose_value)
    # goal += [('AtPose', 12, pose)]
    # goal+=[('On', 1, )]

    goal += [('Holding', a, b) for a, b in problem.goal_holding] + \
                     [('On', b, s) for b, s in problem.goal_on] + \
                     [('Cleaned', b)  for b in problem.goal_cleaned] + \
                     [('Cooked', b)  for b in problem.goal_cooked]

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(problem, collisions=collisions)),
        'sample-grasp': from_list_fn(get_grasp_gen(problem, collisions=False)),
        'inverse-kinematics': from_gen_fn(get_ik_ir_gen(problem, collisions=True, teleport=teleport)),
        'plan-base-motion': from_fn(get_motion_gen(problem, collisions=collisions, teleport=teleport)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=collisions)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(problem, collisions=collisions)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(problem.robot, collisions=collisions)),

        'MoveCost': move_cost_fn,

        # 'TrajPoseCollision': fn_from_constant(False),
        # 'TrajArmCollision': fn_from_constant(False),
        # 'TrajGraspCollision': fn_from_constant(False),
    }
    # get_press_gen(problem, teleport=teleport)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

#######################################################

# TODO: avoid copying this?

def post_process(problem, plan, teleport=False):
    if plan is None:
        return None
    commands = []
    for i, (name, args) in enumerate(plan):
        if name == 'move_base':
            c = args[-1]
            new_commands = c.commands
        elif name == 'pick':
            a, b, p, g, _, c = args
            [t] = c.commands
            close_gripper = GripperCommand(problem.robot, a, g.grasp_width, teleport=teleport)
            attach = Attach(problem.robot, a, g, b)
            new_commands = [t, close_gripper, attach, t.reverse()]
        elif name == 'place':
            a, b, p, g, _, c = args
            [t] = c.commands
            gripper_joint = get_gripper_joints(problem.robot, a)[0]
            position = get_max_limit(problem.robot, gripper_joint)
            open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
            detach = Detach(problem.robot, a, b)
            new_commands = [t, detach, open_gripper, t.reverse()]
        elif name == 'clean': # TODO: add text or change color?
            body, sink = args
            new_commands = [Clean(body)]
        elif name == 'cook':
            body, stove = args
            new_commands = [Cook(body)]
        elif name == 'press_clean':
            body, sink, arm, button, bq, c = args
            [t] = c.commands
            new_commands = [t, Clean(body), t.reverse()]
        elif name == 'press_cook':
            body, sink, arm, button, bq, c = args
            [t] = c.commands
            new_commands = [t, Cook(body), t.reverse()]
        else:
            raise ValueError(name)
        print(i, name, args, new_commands)
        commands += new_commands
    return commands

#######################################################
def pick_problem(arm='right', grasp_type='top', obj_pos=(0., 0, 0)):
    other_arm = get_other_arm(arm)
    initial_conf = get_carry_conf(arm, grasp_type)

    # create pr2 abstract info
    robot_pos=((0.5,0, -2.9348550398988373))

    pr2 = create_pr2()
    set_group_conf(pr2, 'base', robot_pos)
    set_arm_conf(pr2, arm, initial_conf)
    open_arm(pr2, arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)
    mass = 1
    cabbage = create_box(.07, .07, 0.7, color=(0, 1, 0, 1))
    set_point(cabbage, obj_pos)
    
    stock=create_box(.3, .3, 0.05, mass=mass)
    set_point(stock, (1.1,0,-0.1))
    # set_point(cabbage, ((0,0,0)))
    return Problem(pr2, movable=[cabbage], arms=[arm], 
                grasp_types=[grasp_type],
                surfaces=[stock],
                # goal_holding=[(arm, cabbage)],
                goal_on=[(cabbage, stock)]
                )

##########################################################
def main(partial=False, defer=False, verbose=False):
    parser = create_parser()
    parser.add_argument('-cfree', action='store_true', help='Disables collisions during planning')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    print('Arguments:', args)

    connect(use_gui=True)
    set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])
    problem_fn = pick_problem
    # holding_problem | stacking_problem | cleaning_problem | cooking_problem
    # cleaning_button_problem | cooking_button_problem
    with HideOutput():
        problem = problem_fn()
    #state_id = save_state()
    saver = WorldSaver()
    #dump_world()

    yaw = np.random.uniform(*CIRCULAR_LIMITS)
    base_conf = ((1.4,0,yaw))
    set_group_conf(problem.robot, 'base', base_conf)
    success=0
    for i in range(1):
        # set_point(problem.movable[0], (-1+(i%20)*0.1, -i//20*0.1,0))
        
        pddlstream_problem = pddlstream_from_problem(problem, collisions=False, teleport=args.teleport)

        stream_info = {
            # 'test-cfree-pose-pose': StreamInfo(p_success=1e-3, verbose=verbose),
            # 'test-cfree-approach-pose': StreamInfo(p_success=1e-2, verbose=verbose),
            # 'test-cfree-traj-pose': StreamInfo(p_success=1e-1, verbose=verbose),

            'MoveCost': FunctionInfo(opt_move_cost_fn),
        }
        stream_info.update({
            'sample-pose': StreamInfo(opt_gen_fn=PartialInputs('?r'), verbose=verbose),
            'inverse-kinematics': StreamInfo(opt_gen_fn=PartialInputs('?p'), verbose=verbose),
            'plan-base-motion': StreamInfo(opt_gen_fn=PartialInputs('?q1 ?q2'), verbose=verbose, defer_fn=defer_shared if defer else never_defer),
        } if partial else {
            'sample-pose': StreamInfo(opt_gen_fn=from_fn(opt_pose_fn), verbose=verbose),
            'inverse-kinematics': StreamInfo(opt_gen_fn=from_fn(opt_ik_fn), verbose=verbose),
            'plan-base-motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_fn), verbose=verbose),
        })
        _, _, _, stream_map, init, goal = pddlstream_problem
        print('Init:', init)
        print('Goal:', goal)
        print('Streams:', str_from_object(set(stream_map)))
        print(SEPARATOR)

        with Profiler():
            with LockRenderer(lock=not args.enable):
                solution = solve(pddlstream_problem, algorithm=args.algorithm, unit_costs=args.unit,
                                stream_info=stream_info, success_cost=INF, verbose=verbose, debug=False)
                saver.restore()

        # print_solution(solution)
        plan, cost, evaluations = solution
        if (plan is None) or not has_gui():
            print("----------no solution {}-------------".format(i))
            continue

        print(SEPARATOR)
        success+=1
        with LockRenderer(lock=not args.enable):
            commands = post_process(problem, plan)
            problem.remove_gripper()
            saver.restore()

        #restore_state(state_id)
        saver.restore()
        if args.simulate:
            control_commands(commands)
        else:
            apply_commands(State(), commands, time_step=0.01)
        wait_if_gui('Finish?')
    disconnect()
    # TODO: need to wrap circular joints
    print(success)


def plan_pick(robot, body, arm, grasp_type, max_attempts=500, num_samples=1):
    tool_link = get_gripper_link(robot, arm)
    robot_saver = BodySaver(robot)
    gripper_from_base_list = []
    grasps = GET_GRASPS[grasp_type](body)

    start_time = time.time()
    while len(gripper_from_base_list) < num_samples:
        # box_pose = sample_placement(body, table)
        box_pose=Pose((0,0,0.))
        set_pose(body, box_pose)
        grasp_pose = random.choice(grasps)
        gripper_pose = multiply(box_pose, invert(grasp_pose))
        for attempt in range(max_attempts):
            robot_saver.restore()
            base_conf = next(uniform_pose_generator(robot, gripper_pose)) #, reachable_range=(0., 1.)))
            #set_base_values(robot, base_conf)
            set_group_conf(robot, 'base', base_conf)
           
            grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose) #, nearby_conf=USE_CURRENT)
            #conf = inverse_kinematics(robot, link, gripper_pose)
            if (grasp_conf is None) :
                continue
            gripper_from_base = multiply(invert(get_link_pose(robot, tool_link)), get_base_pose(robot))
            #wait_if_gui()
            gripper_from_base_list.append(gripper_from_base)
            print('{} / {} | {} attempts | [{:.3f}]'.format(
                len(gripper_from_base_list), num_samples, attempt, elapsed_time(start_time)))
            # wait_if_gui()
            return grasp_conf
        else:
            print('Failed to find a kinematic solution after {} attempts'.format(max_attempts))
    


def pick_without_move():
    connect(use_gui=True)
    set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])
    arm='right'
    grasp_type="top"
    other_arm = get_other_arm(arm)
    initial_conf = get_carry_conf(arm, grasp_type)
    pr2 = create_pr2()
    set_group_conf(pr2, 'base', ((0.,0,0)))
    set_arm_conf(pr2, arm, initial_conf)
    open_arm(pr2, arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)

    cabbage = create_box(.07, .07, 0.7, mass=1, color=(0, 1, 0, 1))
    set_point(cabbage, (0,-0.5,0))

    yaw = np.random.uniform(*CIRCULAR_LIMITS)
    saver = WorldSaver()

    command=plan_pick(pr2, cabbage, arm, grasp_type)

    disconnect()


def visualize_database():
    cache=load_inverse_reachability("right", "top")
    import matplotlib.pyplot as plt

    x=[]
    y=[]
    for item in cache:
        x.append(item[0][0])
        y.append(item[0][1])
    plt.axis([-1, 1, -1,1])
    plt.plot(x,y, 'go')
    
    plt.show()


def find_closest_base_conf(base_point):
    cache=load_inverse_reachability("right", "top")


def test_pick():
    # box_pose = sample_placement(body, table)
    connect(use_gui=True)
    arm='right'
    grasp_type="top"
    other_arm = get_other_arm(arm)
    pr2 = create_pr2()
    set_group_conf(pr2, 'torso', [0.2])
    set_arm_conf(pr2, arm, get_carry_conf(arm, grasp_type))
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))

    #plane = p.loadURDF("plane.urdf")
    #table = p.loadURDF("table/table.urdf", 0, 0, 0, 0, 0, 0.707107, 0.707107)
    # table = create_table()
    box = create_box(.07, .07, .7)
    box_pose=NewPose((0,0,0))
    set_pose(box, box_pose)
    for i in range(10):
        yaw = np.random.uniform(*CIRCULAR_LIMITS)
        yaw=-2.9348550398988373
        base_conf = ((0.4,0,yaw)) #, reachable_range=(0., 1.)))
        #set_base_values(robot, base_conf)
        
        set_group_conf(pr2, 'base', base_conf)
        grasps = GET_GRASPS[grasp_type](box)
        grasp_pose = random.choice(grasps)
        gripper_pose = multiply(box_pose, invert(grasp_pose))
        grasp_conf = pr2_inverse_kinematics(pr2, arm, gripper_pose) #, nearby_conf=USE_CURRENT)
        #conf = inverse_kinematics(robot, link, gripper_pose)
        if (grasp_conf is None) :
            print("no solution")
        else:
            print("find solution")
            print(yaw)
            break

    disconnect()


def get_ik_gen(problem, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    base_joints = get_group_joints(problem.robot, 'base')
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)
    def gen(*inputs):
        b, a, p, g = inputs
        
        attempts = 0
        while True:
            # yaw_base=-2.9348550398988373
            # yaw = random.random()+yaw_base
            yaw = np.random.uniform(*CIRCULAR_LIMITS)
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                bq=Conf(problem.robot, base_joints, values=(0.5,0,yaw))
                bq.assign()
                ir_outputs = (bq,)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                # print("ik failed")
                continue
            print(yaw)
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            #if not p.init:
            #    return
    return gen

def generate_pick_command():
    connect(use_gui=True)
    set_camera_pose(camera_point=[0,-3,3], target_point=[0,0,0])
    arm='right'
    grasp_type='top'
    max_attempts=100
    box_pose=NewPose((0,0,0))
    problem=pick_problem(arm,grasp_type, obj_pos=box_pose[0])

    placement_gen_fn = get_stable_gen(problem, set_obj=False)
    placement_gen = placement_gen_fn(problem.movable[0], problem.surfaces[0])

    grasp_gen_fn = get_grasp_gen(problem, collisions=True)
    ik_ir_fn = get_ik_gen(problem, max_attempts=max_attempts, learned=False, teleport=True)
    grasps = list(grasp_gen_fn(problem.movable[0]))
    print('Grasps:', len(grasps))

    (g,) = random.choice(grasps)
    box_pose=Pose(problem.movable[0])
    output = next(ik_ir_fn(arm, problem.movable[0], box_pose, g), None)
    if output is None:
        print('Failed to find a solution after {} attempts'.format(max_attempts))
    else:
        (_, ac) = output
        [t,] = ac.commands

        close_gripper = GripperCommand(problem.robot, arm, g.grasp_width, teleport=False)
        attach = Attach(problem.robot, arm, g, problem.movable[0])
        commands = [t, close_gripper, attach, t.reverse()]
        print(commands)
        # apply_commands(State(), commands, time_step=0.01)

    # place
    (p,) = next(placement_gen)
    # place_pose=Pose(problem.movable[0], value=((0.8,0,0),None))
    output = next(ik_ir_fn(arm, problem.movable[0], p, g), None)
    if output is None:
        print('Failed to find a solution after {} attempts'.format(max_attempts))
    else:
        (_, ac) = output
        [t,] = ac.commands
        gripper_joint = get_gripper_joints(problem.robot, arm)[0]
        position = get_max_limit(problem.robot, gripper_joint)
        open_gripper = GripperCommand(problem.robot, arm, position, teleport=False)
        detach = Detach(problem.robot, arm, problem.movable[0])
        new_commands = [t, detach, open_gripper, t.reverse()]

        commands.extend(new_commands)
    apply_commands(State(), commands, time_step=0.01)
    wait_if_gui('Finish?')
    disconnect()

if __name__ == '__main__':
    # pick_without_move()
    # main()
    # visualize_database()
    # test_pick()

    generate_pick_command()
