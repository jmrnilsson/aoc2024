const { timeEnd } = require('node:console');
const {readFile} = require('node:fs/promises');

/*
# ICE
_default_puzzle_input = "year_2022/day_01/puzzle.txt"
_default_test_input = "year_2022/day_01/test.txt"

puzzle_input = build_location(__file__, "puzzle.txt")
test_input = build_location(__file__, "test.txt")
test_input_2 = build_location(__file__, "test_2.txt")
*/

const test_input = "aoc\\year_2022\\day_03_extras_javascript\\test.txt";
const puzzle_input = "aoc\\year_2022\\day_03_extras_javascript\\puzzle.txt";

const challenge_solve_1 = 157
const challenge_solve_2 = 70


function intersect(left, right){
    return new Set([...left].filter(x => right.has(x)));
}

function* chunked(arr, n) {
    for (let i = 0; i < arr.length; i += n) {
        yield arr.slice(i, i + n);
    }
}

/*
test=157
expect=8105
*/ 
async function solve1(input){
    let [seed, value] = [[], 0]

    for (const line of (await readFile(input, { encoding: 'utf8' })).split('\n')){
        if (!line) continue;
        seed.push([line, line.length / 2])
    }

    for (const [rs, len] of seed){
        let [char] = intersect(new Set(rs.split('').slice(0, len)), new Set(rs.split('').slice(len, len*2)));
        value += char == char.toLowerCase() ? char.charCodeAt() - 96 : char.charCodeAt() - 38;
    }

    return value;
}

/*
test=70
expect=2363
*/ 
async function solve2(input){
    let [seed, value] = [[], 0]

    const lines = (await readFile(input, { encoding: 'utf8' })).split('\n');
    for (const chunk of chunked(lines.slice(0, lines.length -2), 3)){
        seed.push([chunk])
    }

    for (const [rucksack] of seed){
        var [tail, head] = [rucksack.slice(1, rucksack.length), new Set(rucksack[0])];
        let [char] = tail.reduce((a, v) => intersect(a, new Set(v)), head);
        value += char == char.toLowerCase() ? char.charCodeAt() - 96 : char.charCodeAt() - 38;
    }

    return value;
}

async function timed(func_name, ...args){
    const func = eval(func_name);
    ts = performance.now();
    const answer = await func(...args)
    ms = (performance.now() - ts).toFixed(4);
    const name = args[0].match("(test|puzzle)", 'i')[1];
    return `${func_name} - ${name.padEnd(6," ")} (${ms}ms):`.padEnd(33," ") + answer;
}


async function start(argv) {
    console.log(await timed("solve1", test_input));
    console.log(await timed("solve1", puzzle_input));
    console.log(await timed("solve2", test_input));
    console.log(await timed("solve2", puzzle_input));
    console.log("\nDone!");
    process.exit();
}

start(process.argv);
