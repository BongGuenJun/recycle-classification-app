import { Detection } from './types';

const materialNames:Record<string,string>={can:'캔',pet:'페트병',plastic:'플라스틱'};
const conditionNames:Record<string,string>={clean:'오염 없음',outer:'외부 오염',inner:'내부 오염'};
const conditionActions:Record<string,string>={clean:'깨끗한 상태예요',outer:'외부 오염을 제거해주세요',inner:'내부를 깨끗이 헹궈주세요'};
const disposalSteps:Record<string,string[]>={
  can:['내용물을 완전히 비워주세요.','표면의 이물질을 제거하고 헹궈주세요.','다른 재질의 뚜껑과 부품은 분리해주세요.'],
  pet:['내용물을 완전히 비워주세요.','라벨과 뚜껑을 분리하고 깨끗이 헹궈주세요.','가능하면 압착한 뒤 지역 배출 기준에 맞춰주세요.'],
  plastic:['내용물을 완전히 비워주세요.','이물질을 제거하고 깨끗이 헹궈주세요.','다른 재질의 뚜껑과 부품은 분리해주세요.'],
};

export function getResultCopy(value:Pick<Detection,'class_name'|'material'|'dirtiness'>){
  const parts=(value.class_name||'').split('_');const material=value.material||parts[0]||'';const dirtiness=value.dirtiness||parts[1]||'';
  return{material, dirtiness, materialName:materialNames[material]||'재활용품', conditionName:conditionNames[dirtiness]||'상태 확인', action:conditionActions[dirtiness]||'배출 전 상태를 확인해주세요', steps:disposalSteps[material]||['내용물을 비워주세요.','이물질과 다른 재질을 제거해주세요.','지역별 배출 기준을 확인해주세요.']};
}
