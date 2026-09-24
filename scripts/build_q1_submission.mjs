import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";


async function importWorkbook(filePath) {
  return SpreadsheetFile.importXlsx(await FileBlob.load(filePath));
}


async function renderQ1(workbook, previewPath) {
  const preview = await workbook.render({
    sheetName: "Q1_单点组批",
    range: "A1:I22",
    scale: 2,
    format: "png",
  });
  await fs.mkdir(path.dirname(previewPath), { recursive: true });
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
}


async function inspectTemplate(templatePath, previewPath) {
  const workbook = await importWorkbook(templatePath);
  const sheets = await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 4000,
  });
  const table = await workbook.inspect({
    kind: "table",
    sheetId: "Q1_单点组批",
    range: "A1:I6",
    include: "values,formulas",
    tableMaxRows: 6,
    tableMaxCols: 9,
    maxChars: 6000,
  });
  const styles = await workbook.inspect({
    kind: "computedStyle",
    sheetId: "Q1_单点组批",
    range: "A1:I3",
    maxChars: 6000,
  });
  console.log(sheets.ndjson);
  console.log(table.ndjson);
  console.log(styles.ndjson);
  await renderQ1(workbook, previewPath);
}


async function buildSubmission(templatePath, csvPath, outputPath, previewPath) {
  const workbook = await importWorkbook(templatePath);
  const csvText = await fs.readFile(csvPath, "utf8");
  const csvWorkbook = await Workbook.fromCSV(csvText, { sheetName: "Trips" });
  const csvSheet = csvWorkbook.worksheets.getItem("Trips");
  const sourceValues = csvSheet.getUsedRange(true).values;
  const rows = sourceValues.slice(1).filter((row) => row[0] !== null && row[0] !== "");
  if (rows.length !== 18) {
    throw new Error(`expected 18 recommended trips, found ${rows.length}`);
  }
  const typedRows = rows.map((row) => [
    String(row[0]),
    String(row[1]),
    String(row[2]),
    String(row[3]),
    Number(row[4]),
    Number(row[5]),
    Number(row[6]),
    Number(row[7]),
    Number(row[8]),
  ]);
  if (typedRows.some((row) => row.slice(4).some((value) => !Number.isFinite(value)))) {
    throw new Error("recommended trip CSV contains a non-numeric output value");
  }

  const sheet = workbook.worksheets.getItem("Q1_单点组批");
  sheet.getRange("A2:I19").values = typedRows;
  sheet.getRange("E2:E19").format.numberFormat = "0.0";
  sheet.getRange("F2:F19").format.numberFormat = "0.000";
  sheet.getRange("G2:G19").format.numberFormat = "0.00";
  sheet.getRange("H2:H19").format.numberFormat = "0.000000";
  sheet.getRange("I2:I19").format.numberFormat = "0.0000";
  sheet.getRange("D2:D19").format.wrapText = true;
  sheet.getRange("A2:I19").format.autofitRows();
  workbook.recalculate();

  const table = await workbook.inspect({
    kind: "table",
    sheetId: "Q1_单点组批",
    range: "A1:I19",
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 9,
    maxChars: 16000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });
  console.log(table.ndjson);
  console.log(errors.ndjson);

  await renderQ1(workbook, previewPath);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const saved = await importWorkbook(outputPath);
  const savedTable = await saved.inspect({
    kind: "table",
    sheetId: "Q1_单点组批",
    range: "A1:I19",
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 9,
    maxChars: 16000,
  });
  console.log(savedTable.ndjson);
}


const [mode, templatePath, csvPathOrPreview, outputPath, previewPath] = process.argv.slice(2);
if (mode === "inspect") {
  if (!templatePath || !csvPathOrPreview) {
    throw new Error("usage: inspect <template.xlsx> <preview.png>");
  }
  await inspectTemplate(templatePath, csvPathOrPreview);
} else if (mode === "build") {
  if (!templatePath || !csvPathOrPreview || !outputPath || !previewPath) {
    throw new Error("usage: build <template.xlsx> <trips.csv> <output.xlsx> <preview.png>");
  }
  await buildSubmission(templatePath, csvPathOrPreview, outputPath, previewPath);
} else {
  throw new Error(`unknown mode: ${mode}`);
}
