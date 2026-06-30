import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { faker } from "@faker-js/faker/locale/en_IN";

const OUT_PATH = "eval/bench/data/tamil.jsonl";

interface PrivacyMask {
  label: string;
  start: number;
  end: number;
  value: string;
}

interface Row {
  uid: number;
  language: string;
  source_text: string;
  privacy_mask: PrivacyMask[];
}

let nextUid = 1000000;

function createRow(
  text: string,
  language: string,
  entities: { value: string; label: string }[]
): Row {
  const mask: PrivacyMask[] = [];
  for (const ent of entities) {
    const start = text.indexOf(ent.value);
    if (start !== -1) {
      mask.push({
        label: ent.label,
        start,
        end: start + ent.value.length,
        value: ent.value,
      });
    }
  }
  return {
    uid: nextUid++,
    language,
    source_text: text,
    privacy_mask: mask,
  };
}

const TA_GIVEN = [
  "அர்ஜுன்", "முகிலன்", "கவின்", "சூர்யா", "கார்த்திக்", "விக்ரம்", "விஜய்", "தனுஷ்",
  "நயன்தாரா", "த்ரிஷா", "சினேகா", "ரம்யா", "பிரியா", "அனிதா", "திவ்யா"
];
const TA_SURNAME = [
  "ராமசாமி", "கிருஷ்ணன்", "சுப்ரமணியன்", "நடராஜன்", "பாலகிருஷ்ணன்", "ராதாகிருஷ்ணன்",
  "சீனிவாசன்", "வெங்கடேசன்", "ரங்கநாதன்", "ராஜன்", "ராஜகோபால்", "மாணிக்கம்", "கந்தசாமி",
  "முருகேசன்", "கருப்பையா"
];

const EN_GIVEN = [
  "Arjun", "Muhilan", "Kavin", "Surya", "Karthik", "Vikram", "Vijay", "Dhanush",
  "Nayanthara", "Trisha", "Sneha", "Ramya", "Priya", "Anitha", "Divya"
];
const EN_SURNAME = [
  "Ramasamy", "Krishnan", "Subramanian", "Natarajan", "Balakrishnan", "Radhakrishnan",
  "Srinivasan", "Venkatesan", "Ranganathan", "Rajan", "Rajagopal", "Manickam", "Kandasamy",
  "Murugesan", "Karuppiah"
];

function sample(arr: string[]): string {
  return arr[Math.floor(Math.random() * arr.length)];
}

function generateCases(useRomanized: boolean): Row[] {
  const f = faker; // For addresses/phones we just use en_IN format (realistic Indian)
  const lang = useRomanized ? "ta_Latn" : "ta";
  const givens = useRomanized ? EN_GIVEN : TA_GIVEN;
  const surnames = useRomanized ? EN_SURNAME : TA_SURNAME;
  
  const rows: Row[] = [];
  const numPerTemplate = 25; // 5 templates * 25 = 125 cases
  
  for (let i = 0; i < numPerTemplate; i++) {
    const first = sample(givens);
    const last = sample(surnames);
    const text = `My name is ${first} ${last} and I need help with my application.`;
    rows.push(createRow(text, lang, [
      { value: first, label: "GIVENNAME" },
      { value: last, label: "SURNAME" }
    ]));
  }
  
  for (let i = 0; i < numPerTemplate; i++) {
    const first = sample(givens);
    const last = sample(surnames);
    const text = `Please update the record for ${first} ${last}.`;
    rows.push(createRow(text, lang, [
      { value: first, label: "GIVENNAME" },
      { value: last, label: "SURNAME" }
    ]));
  }
  
  for (let i = 0; i < numPerTemplate; i++) {
    const first = sample(givens);
    const last = sample(surnames);
    const text = `The applicant is ${first} ${last} and the household size is four.`;
    rows.push(createRow(text, lang, [
      { value: first, label: "GIVENNAME" },
      { value: last, label: "SURNAME" }
    ]));
  }
  
  for (let i = 0; i < numPerTemplate; i++) {
    const first = sample(givens);
    const last = sample(surnames);
    const phone = f.phone.number({ style: 'national' });
    const text = `I am ${first} ${last} and my phone number is ${phone}.`;
    rows.push(createRow(text, lang, [
      { value: first, label: "GIVENNAME" },
      { value: last, label: "SURNAME" },
      { value: phone, label: "TELEPHONENUM" }
    ]));
  }
  
  for (let i = 0; i < numPerTemplate; i++) {
    const first = sample(givens);
    const last = sample(surnames);
    const building = String(f.number.int({ min: 1, max: 999 }));
    const street = f.location.street();
    const text = `${first} ${last} is listed as the head of household, living at ${building} ${street}.`;
    rows.push(createRow(text, lang, [
      { value: first, label: "GIVENNAME" },
      { value: last, label: "SURNAME" },
      { value: building, label: "BUILDINGNUM" },
      { value: street, label: "STREET" }
    ]));
  }
  
  return rows;
}

async function main() {
  await mkdir(dirname(OUT_PATH), { recursive: true });
  
  const tamilCases = generateCases(false);
  const romanizedCases = generateCases(true);
  
  const allCases = [...tamilCases, ...romanizedCases];
  
  let content = "";
  for (const row of allCases) {
    content += JSON.stringify(row) + "\n";
  }
  
  await writeFile(OUT_PATH, content);
  console.log(`Generated ${allCases.length} cases to ${OUT_PATH}`);
}

main().catch(console.error);
