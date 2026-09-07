import { Directory, File, Paths } from 'expo-file-system';

const sessionDirectory = () => new Directory(Paths.cache, 'recycle-analysis-session');

export function resetSessionFiles(){
  try{const directory=sessionDirectory();if(directory.exists)directory.delete();directory.create({idempotent:true,intermediates:true});}catch{/* 캐시 정리는 다음 실행에서 다시 시도 */}
}

export function keepSessionCrop(temporaryUri:string,id:string){
  const directory=sessionDirectory();if(!directory.exists)directory.create({idempotent:true,intermediates:true});
  const source=new File(temporaryUri);const destination=new File(directory,`${id}.jpg`);source.move(destination);return destination.uri;
}
