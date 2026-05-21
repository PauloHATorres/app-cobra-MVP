const URL_CLASSIFICADOR = "./modelo_tfjs/";
const URL_DETECTOR = "./modelo_detector/model.json";

let modeloDetector;
let model;

let ultimaAnalisePrecisaElapidico = false;
let ultimaPosicaoUsuario = null;

const LIMIAR_DETECTOR_COBRA = 0.25;
const LIMIAR_CLASSIFICADOR_INCERTO = 0.60;
const LIMIAR_RISCO_ALTO = 0.60;
const LIMIAR_OUTRAS_COBRAS = 0.70;
const LIMIAR_CORAL = 0.50;

const centros = [
  {
    nome: "HC UNESP Botucatu",
    cidade: "Botucatu",
    telefone: "(14) 3811-6129",
    latitude: -22.8902815,
    longitude: -48.4940742,
    maps: "https://www.google.com/maps/place/Pronto-Socorro+Referenciado+HC+Unesp/@-22.8902815,-48.4940742,1006m/data=!3m1!1e3!4m6!3m5!1s0x94c7275e589095e7:0xe3229bcaf9cb7c08!8m2!3d-22.88909!4d-48.4968378!16s%2Fg%2F11ddydq4_b?entry=ttu&g_ep=EgoyMDI2MDUxNy4wIKXMDSoASAFQAw%3D%3D",
    soros: ["Botrópico", "Crotálico", "Elapídico", "Laquético", "Escorpiônico", "Loxoscélico", "Fonêutrico", "Lonômico"]
  },
  {
    nome: "Santa Casa de Misericórdia",
    cidade: "Avaré",
    telefone: "(14) 3711-9100",
    latitude: -23.1011996,
    longitude: -48.9311197,
    maps: "https://www.google.com/maps/place/Santa+Casa+de+Miseric%C3%B3rdia/@-23.1011996,-48.9311197,1004m/data=!3m2!1e3!4b1!4m6!3m5!1s0x94c12685022dc38b:0x4b2525719a98ab0b!8m2!3d-23.1012046!4d-48.9285448!16s%2Fg%2F1ptx3k_d4?entry=ttu&g_ep=EgoyMDI2MDUxNy4wIKXMDSoASAFQAw%3D%3D",
    soros: ["Botrópico", "Crotálico", "Loxoscélico", "Fonêutrico", "Escorpiônico"]
  },
  {
    nome: "Pronto Socorro Municipal Central (PSMC)",
    cidade: "Bauru",
    telefone: "(14) 3104-1160 / (14) 3104-1166",
    latitude: -22.3310073,
    longitude: -49.0797829,
    maps: "https://www.google.com/maps/place/Pronto+Socorro+Central/@-22.3310073,-49.0797829,714m/data=!3m1!1e3!4m6!3m5!1s0x94bf67afcc8d02ad:0xaa99d1ac014ce9e5!8m2!3d-22.3308275!4d-49.0780593!16s%2Fg%2F11g234x0yq?entry=ttu&g_ep=EgoyMDI2MDUxNy4wIKXMDSoASAFQAw%3D%3D",
    soros: ["Botrópico", "Crotálico", "Escorpiônico", "Lonômico", "Loxoscélico", "Fonêutrico"]
  },
  {
    nome: "UPA Bela Vista",
    cidade: "Bauru",
    telefone: "(14) 3102-1213 / 1212 / 1214 / 1234",
    latitude: -22.3087208,
    longitude: -49.0922473,
    maps: "https://www.google.com/maps/place/UPA+Bela+Vista/@-22.3087208,-49.0922473,1010m/data=!3m2!1e3!4b1!4m6!3m5!1s0x94bf6711ddf19473:0x48a6be42874a38a!8m2!3d-22.3087258!4d-49.0896724!16s%2Fg%2F11b6t4yc4c?entry=ttu&g_ep=EgoyMDI2MDUxNy4wIKXMDSoASAFQAw%3D%3D",
    soros: ["Botrópico", "Crotálico", "Elapídico", "Escorpiônico", "Loxoscélico", "Fonêutrico"]
  }
];

async function carregarModelo() {
  try {
    document.getElementById("resultado").innerText =
      "Carregando modelos...";

    modeloDetector = await tf.loadGraphModel(URL_DETECTOR);

    model = await tf.loadLayersModel(
      URL_CLASSIFICADOR + "model.json"
    );

    console.log("Detector YOLO carregado!");
    console.log("Classificador antigo carregado!");

    document.getElementById("resultado").innerText =
      "Envie uma imagem para analisar.";

  } catch (erro) {
    console.error("Erro ao carregar modelos:", erro);

    document.getElementById("resultado").innerText =
      "Erro ao carregar o modelo.";
  }
}

async function detectarCobra(imagem) {
  if (!modeloDetector) return false;

  const tensor = tf.browser.fromPixels(imagem)
    .resizeNearestNeighbor([640, 640])
    .toFloat()
    .div(255)
    .expandDims();

  let saida;
  let temCobra = false;
  let maiorConfianca = 0;

  try {
    saida = await modeloDetector.executeAsync(tensor);

    let tensorDeteccoes = null;

    if (Array.isArray(saida)) {
      tensorDeteccoes = saida.find(t => {
        const shape = t.shape;
        return shape.length === 3 && shape[2] === 6;
      }) || saida[0];
    } else if (saida["Identity:0"]) {
      tensorDeteccoes = saida["Identity:0"];
    } else {
      const valores = Object.values(saida);
      tensorDeteccoes = valores.find(t => {
        const shape = t.shape;
        return shape.length === 3 && shape[2] === 6;
      }) || valores[0];
    }

    const dados = await tensorDeteccoes.data();

    for (let i = 0; i < dados.length; i += 6) {
      const confianca = dados[i + 4];

      if (confianca > maiorConfianca) {
        maiorConfianca = confianca;
      }

      if (confianca >= LIMIAR_DETECTOR_COBRA) {
        temCobra = true;
        break;
      }
    }

    console.log("Maior confiança do detector:", maiorConfianca);
    console.log("Detector encontrou cobra?", temCobra);

  } catch (erro) {
    console.error("Erro ao rodar detector:", erro);
    temCobra = false;
  } finally {
    tensor.dispose();
    if (saida) tf.dispose(saida);
  }

  return temCobra;
}

async function analisarImagem(imagem) {

  if (!model || !modeloDetector) {
    document.getElementById("resultado").innerText =
      "Modelo ainda carregando...";
    return;
  }

  document.getElementById("resultado").innerText =
    "Analisando imagem...";

  const temCobra = await detectarCobra(imagem);

  if (!temCobra) {
    const classificacao =
      "🟢 NÃO PARECE SER SERPENTE";

    const recomendacao =
      "A imagem não parece conter uma serpente. Ainda assim, evite contato com animais desconhecidos.";

    ultimaAnalisePrecisaElapidico = false;
    esconderCentrosNaAnalise();

    document.getElementById("resultado").innerText =
      `${classificacao}\n\n${recomendacao}`;

    return;
  }

  const tensor = tf.browser.fromPixels(imagem)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .sub(127.5)
    .div(127.5)
    .expandDims();

  const pred = await model.predict(tensor).data();

  const classNames = [
    "cascavel",
    "coral",
    "jararaca",
    "nao_cobra",
    "outras_serpentes"
  ];

  const probs = {};

  classNames.forEach((nome, i) => {
    probs[nome] = pred[i];
  });

  const jararaca = probs["jararaca"] || 0;
  const cascavel = probs["cascavel"] || 0;
  const coral = probs["coral"] || 0;
  const outrasCobras = probs["outras_serpentes"] || 0;
  const naoCobra = probs["nao_cobra"] || 0;

  const maiorPeconhenta = Math.max(
    jararaca,
    cascavel,
    coral
  );

  const classeMaisProvavel =
    classNames[pred.indexOf(Math.max(...pred))];

  const possivelCoral =
    coral >= LIMIAR_CORAL ||
    (classeMaisProvavel === "coral" && coral >= 0.30);

  let classificacao = "";
  let recomendacao = "";

  if (possivelCoral) {

    classificacao =
      "🔴 ALTO RISCO DE SERPENTE PEÇONHENTA";

    recomendacao =
      "Não tente manusear o animal. Em caso de mordida ou contato suspeito, procure atendimento médico imediatamente.\n\n" +
      "⚠️ A imagem apresentou compatibilidade com grupo que pode exigir soro elapídico. Por isso, o sistema priorizou unidades com esse soro.";

    ultimaAnalisePrecisaElapidico = true;

    mostrarBlocoCentros(true);

  } else if (maiorPeconhenta >= LIMIAR_RISCO_ALTO) {

    classificacao =
      "🔴 ALTO RISCO DE SERPENTE PEÇONHENTA";

    recomendacao =
      "Não tente manusear o animal. Em caso de mordida ou contato suspeito, procure atendimento médico imediatamente.";

    ultimaAnalisePrecisaElapidico = false;

    mostrarBlocoCentros(false);

  } else if (outrasCobras >= LIMIAR_OUTRAS_COBRAS) {

    classificacao =
      "🟡 SERPENTE DETECTADA — RISCO INCERTO";

    recomendacao =
      "O sistema detectou uma serpente, mas não confirmou grupo de alto risco. Não manuseie. Em caso de mordida, procure atendimento médico o quanto antes.";

    ultimaAnalisePrecisaElapidico = false;

    mostrarBlocoCentros(false);

  } else if (naoCobra >= LIMIAR_CLASSIFICADOR_INCERTO) {

    classificacao =
      "🟡 SERPENTE DETECTADA — CLASSIFICAÇÃO INCERTA";

    recomendacao =
      "O detector encontrou uma possível serpente na imagem, mas o classificador antigo não conseguiu confirmar o grupo com segurança. Mantenha distância e, em caso de mordida ou contato suspeito, procure atendimento médico.";

    ultimaAnalisePrecisaElapidico = false;

    mostrarBlocoCentros(false);

  } else {

    classificacao =
      "⚠️ RESULTADO INCONCLUSIVO";

    recomendacao =
      "A imagem parece conter uma serpente, mas não permitiu uma classificação confiável. Tente outra foto com melhor iluminação e distância segura.";

    ultimaAnalisePrecisaElapidico = false;

    mostrarBlocoCentros(false);
  }

  document.getElementById("resultado").innerText =
    `${classificacao}\n\n${recomendacao}`;

  tensor.dispose();
}

function mostrarBlocoCentros(apenasElapidico) {
  const bloco = document.getElementById("orientacaoCentros");
  const mensagem = document.getElementById("mensagemCentros");

  bloco.classList.remove("oculto");

  if (apenasElapidico) {
    mensagem.innerText = "Imagem compatível com possível coral. Serão priorizadas unidades com soro elapídico.";
  } else {
    mensagem.innerText = "Em caso de acidente ou contato suspeito, procure atendimento médico. Use sua localização para ver os centros mais próximos.";
  }

  renderizarCentros("listaCentrosAnalise", apenasElapidico, ultimaPosicaoUsuario);
}

function esconderCentrosNaAnalise() {
  document.getElementById("orientacaoCentros").classList.add("oculto");
  document.getElementById("listaCentrosAnalise").innerHTML = "";
}

function obterLocalizacao(callback) {
  if (!navigator.geolocation) {
    alert("Seu navegador não permite acesso à localização.");
    callback(null);
    return;
  }

  navigator.geolocation.getCurrentPosition(
    pos => {
      ultimaPosicaoUsuario = {
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude
      };
      callback(ultimaPosicaoUsuario);
    },
    erro => {
      console.error("Erro de localização:", erro);
      alert("Não foi possível acessar sua localização. Verifique a permissão do navegador.");
      callback(null);
    },
    {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 60000
    }
  );
}

function calcularDistanciaKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = grausParaRad(lat2 - lat1);
  const dLon = grausParaRad(lon2 - lon1);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(grausParaRad(lat1)) * Math.cos(grausParaRad(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function grausParaRad(graus) {
  return graus * Math.PI / 180;
}

function prepararCentros(apenasElapidico, posicaoUsuario) {
  let lista = [...centros];

  if (apenasElapidico) {
    lista = lista.filter(centro => centro.soros.includes("Elapídico"));
  }

  if (posicaoUsuario) {
    lista = lista.map(centro => ({
      ...centro,
      distancia: calcularDistanciaKm(
        posicaoUsuario.latitude,
        posicaoUsuario.longitude,
        centro.latitude,
        centro.longitude
      )
    })).sort((a, b) => a.distancia - b.distancia);
  }

  return lista;
}

function renderizarCentros(elementoId, apenasElapidico = false, posicaoUsuario = null) {
  const container = document.getElementById(elementoId);
  const lista = prepararCentros(apenasElapidico, posicaoUsuario);

  if (lista.length === 0) {
    container.innerHTML = "<p class='texto-apoio'>Nenhum centro encontrado para esse filtro.</p>";
    return;
  }

  container.innerHTML = lista.map(centro => {
    const distancia = centro.distancia !== undefined
      ? `<p class="distancia">${centro.distancia.toFixed(1)} km de distância aproximada</p>`
      : "";

    const rota = `https://www.google.com/maps/dir/?api=1&destination=${centro.latitude},${centro.longitude}`;
    const telefoneLink = centro.telefone.split("/")[0].replace(/[^0-9]/g, "");

    return `
      <article class="centro-card">
        <div>
          <h3>${centro.nome}</h3>
          <p class="cidade">${centro.cidade}</p>
          ${distancia}
          <p><strong>Telefone:</strong> ${centro.telefone}</p>
          <p><strong>Soros:</strong> ${centro.soros.join(", ")}</p>
        </div>
        <div class="acoes-centro">
          <a href="${rota}" target="_blank" rel="noopener">Abrir rota</a>
          <a href="tel:${telefoneLink}">Ligar</a>
          <a href="${centro.maps}" target="_blank" rel="noopener">Ver no Maps</a>
        </div>
      </article>
    `;
  }).join("");
}

function trocarAba(nomeAba) {
  document.querySelectorAll(".aba").forEach(botao => {
    botao.classList.toggle("ativa", botao.dataset.aba === nomeAba);
  });

  document.querySelectorAll(".conteudo-aba").forEach(secao => {
    secao.classList.remove("ativo");
  });

  document.getElementById(`aba-${nomeAba}`).classList.add("ativo");
}

document.querySelectorAll(".aba").forEach(botao => {
  botao.addEventListener("click", () => trocarAba(botao.dataset.aba));
});

document.getElementById("imagemInput").addEventListener("change", function(event) {
  const arquivo = event.target.files[0];
  if (!arquivo) return;

  const img = document.getElementById("preview");

  img.onload = async function() {
    console.log("Imagem carregada!");
    await analisarImagem(img);
  };

  img.src = URL.createObjectURL(arquivo);
  img.style.display = "block";
});

document.getElementById("btnLocalizacaoAnalise").addEventListener("click", () => {
  obterLocalizacao(posicao => {
    renderizarCentros("listaCentrosAnalise", ultimaAnalisePrecisaElapidico, posicao);
  });
});

document.getElementById("btnLocalizacaoCentros").addEventListener("click", () => {
  obterLocalizacao(posicao => {
    const apenasElapidico = document.getElementById("filtroElapidico").checked;
    renderizarCentros("listaCentrosGeral", apenasElapidico, posicao);
  });
});

document.getElementById("btnMostrarTodos").addEventListener("click", () => {
  document.getElementById("filtroElapidico").checked = false;
  renderizarCentros("listaCentrosGeral", false, ultimaPosicaoUsuario);
});

document.getElementById("filtroElapidico").addEventListener("change", event => {
  renderizarCentros("listaCentrosGeral", event.target.checked, ultimaPosicaoUsuario);
});

renderizarCentros("listaCentrosGeral", false, null);
carregarModelo();