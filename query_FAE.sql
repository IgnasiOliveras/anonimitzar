SELECT
  x.*,  -- All columns from xats
  t.*   -- All columns from tipificaciones
FROM `bbdd1-453614.bd_xats.xats` x
LEFT JOIN `bbdd1-453614.bd_xats.tipificaciones` t 
  ON x.OpenchannelInteractionId = t.id 
  AND x.ContactId = t.ContactId;

WITH merged1 AS (
  SELECT 
    *, 
    FORMAT_DATE('%d/%m/%Y', DATE(createdAt)) AS fecha1,
    EXTRACT(HOUR FROM createdAt) AS hora1,
    DENSE_RANK() OVER (
      PARTITION BY ContactId, DATE(createdAt) 
      ORDER BY createdAt
    ) AS conversacion_dia1
  FROM bbdd1-453614.bd_xats.merged1
), 

fichas AS (
  SELECT
    *, 
    FORMAT_DATE('%d/%m/%Y', DATE(Data_hora_inici)) AS fecha2,
    EXTRACT(HOUR FROM Data_hora_inici) AS hora2,
    DENSE_RANK() OVER (
      PARTITION BY ContactId, DATE(Data_hora_inici) 
      ORDER BY Data_hora_inici
    ) AS conversacion_dia2
  FROM bbdd1-453614.bd_xats.fichas
)

SELECT 
  m.*, 
  f.*
FROM merged1 AS m
FULL JOIN fichas AS f
ON m.ContactId = f.ContactId
AND m.fecha1 = f.fecha2
AND m.conversacion_dia1 = f.conversacion_dia2
ORDER BY m.id_x, m.ContactId;
