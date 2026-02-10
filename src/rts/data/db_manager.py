import oracledb
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from ..config.config_manager import DBConfig

logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self, config: DBConfig):
        self.config = config
        self.conn = None

    def _get_connection(self):
        if not self.conn or not self.conn.is_healthy():
            try:
                self.conn = oracledb.connect(
                    user=self.config.user,
                    password=self.config.password,
                    dsn=self.config.dsn
                )
                logger.info("Successfully connected to OracleDB")
            except Exception as e:
                logger.error(f"Failed to connect to OracleDB: {e}")
                raise
        return self.conn

    def fetch_data(self, rule_timekey: str) -> Dict[str, Any]:
        """Fetch all necessary data for a specific RULE_TIMEKEY."""
        conn = self._get_connection()
        data = {}
        
        queries = {
            "capabilities": f"SELECT PRODUCT, PROCESS, MODEL, ST, FEASIBLE, INITIAL_COUNT FROM RTS_EQP_CAPA_INF WHERE RULE_TIMEKEY = :tk",
            "changeover": f"SELECT FROM_PRODUCT, FROM_PROCESS, TO_PRODUCT, TO_PROCESS, CO_TIME, DEFAULT_TIME FROM RTS_CO_RULE_INF WHERE RULE_TIMEKEY = :tk",
            "inventory": f"SELECT MODEL, CNT AS count FROM RTS_EQP_INV_INF WHERE RULE_TIMEKEY = :tk",
            "plan_wip": f"SELECT PRODUCT, PROCESS, OPER_SEQ, WIP, PLAN FROM RTS_PLAN_WIP_INF WHERE RULE_TIMEKEY = :tk",
            "downtime": f"SELECT MODEL, START_STEP, END_STEP, CNT AS count FROM RTS_EQP_DT_INF WHERE RULE_TIMEKEY = :tk"
        }
        
        with conn.cursor() as cursor:
            # 1. Capabilities
            cursor.execute(queries["capabilities"], tk=rule_timekey)
            columns = [col[0].lower() for col in cursor.description]
            data["capabilities"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
            for cap in data["capabilities"]:
                cap['feasible'] = True if cap['feasible'] == 'Y' else False
                # Normalize Oracle numeric types to Python native types
                # Oracle may return Decimal or other special types that behave
                # differently from Python float/int during downstream processing
                cap['st'] = float(cap['st'])
                cap['initial_count'] = int(cap['initial_count'])

            # 2. Changeover
            cursor.execute(queries["changeover"], tk=rule_timekey)
            rows = cursor.fetchall()
            if rows:
                data["changeover"] = {
                    "default_time": float(rows[0][5]),
                    "rules": [dict(zip(
                        ["from_product", "from_process", "to_product", "to_process", "time"],
                        [str(row[0]), str(row[1]), str(row[2]), str(row[3]), float(row[4])]
                    )) for row in rows]
                }
            else:
                data["changeover"] = {"default_time": 60.0, "rules": []}

            # 3. Inventory
            cursor.execute(queries["inventory"], tk=rule_timekey)
            columns = [col[0].lower() for col in cursor.description]
            data["inventory"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
            for inv in data["inventory"]:
                inv['count'] = int(inv['count'])

            # 4. Plan WIP
            cursor.execute(queries["plan_wip"], tk=rule_timekey)
            columns = [col[0].lower() for col in cursor.description]
            data["plan_wip"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
            for pw in data["plan_wip"]:
                pw['oper_seq'] = int(pw['oper_seq'])
                pw['wip'] = float(pw['wip'])
                pw['plan'] = float(pw['plan'])

            # 5. Downtime
            cursor.execute(queries["downtime"], tk=rule_timekey)
            columns = [col[0].lower() for col in cursor.description]
            data["downtime"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
            for dt in data["downtime"]:
                dt['start_step'] = int(dt['start_step'])
                dt['end_step'] = int(dt['end_step'])
                dt['count'] = int(dt['count'])
            
        return data

    def upload_results(self, rule_timekey: str, results: List[Dict[str, Any]]):
        """Upload inference results to RTS_RESLT_INF."""
        conn = self._get_connection()
        sql = """
            INSERT INTO RTS_RESLT_INF (
                RULE_TIMEKEY, SIM_STEP, PRODUCT, PROCESS, WIP, PRODUCTION, 
                ACTIVE_EQP, TARGET_EQP, UNAVAILABLE_EQP, PLAN, PRODUCED_SUM, TOTAL_CO
            ) VALUES (
                :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12
            )
        """
        rows = []
        for r in results:
            rows.append((
                rule_timekey, r['timestamp'], r['product'], r['process'],
                float(r['wip']), float(r['production']), float(r['active_eqp']),
                float(r['target_eqp']), float(r.get('unavailable_eqp', 0)),
                float(r['plan']), float(r['produced_sum']), int(r['total_changeovers'])
            ))
            
        with conn.cursor() as cursor:
            # 1. Clear existing results for this timekey
            cursor.execute("DELETE FROM RTS_RESLT_INF WHERE RULE_TIMEKEY = :tk", tk=rule_timekey)
            
            # 2. Batch insert new results
            cursor.executemany(sql, rows)
            conn.commit()
        logger.info(f"Successfully uploaded {len(rows)} result rows for {rule_timekey}")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
